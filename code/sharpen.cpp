#include <unistd.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <iomanip> 
#include "pixel.h"

using namespace std;
using namespace cv;

std::string inputName = "";
double img_percent_threshold = 0.025;
bool morph_gradient = false;
bool opening_gradient = false;
bool invertColors = false;
int num_iterations = 1;

struct img_with_count {
    imageGrid img;
    int object_count;
};

double blur_list[] = {  1, 2, 1,
                        2, 4, 2,
                        1, 2, 1  };

double sharp_list[] = {  -1,  -2, -1,
                         -2,  13, -2,
                         -1,  -2, -1  };

double morph_list[] = {  0,  2,  0,
                         2,  8,  2,
                         0,  2,  0  };

mask   blur(sqrt(sizeof(blur_list)  / sizeof(double)), sizeof(blur_list)  / sizeof(double),  blur_list, 1 / 16.0);
mask  sharp(sqrt(sizeof(sharp_list) / sizeof(double)), sizeof(sharp_list) / sizeof(double), sharp_list, 1 /  1.0);
mask  morph(sqrt(sizeof(morph_list) / sizeof(double)), sizeof(morph_list) / sizeof(double), morph_list, 1 /  1.0);

string toString(bool b) {
    return (b ? "TRUE" : "FALSE");
}

// img will exit holding the image before clustering,
// the returned object will hold an the clustered image
img_with_count count_objects(imageGrid& img) {
    img.toGrey();
    for (int j = 0; j < num_iterations; j++) {
        imageGrid gradient_img = img;
        imageGrid closing = img;
        imageGrid opening = img;

        if (morph_gradient) {
            gradient_img = img;
            gradient_img.morph_gradient(morph); //get morph gradient
            img.subtract(gradient_img);         //remove morph gradient
        }

        if (opening_gradient) {
            closing.closing(morph);
            opening.opening(morph);
            closing.subtract(opening);
            img.subtract(closing);

            img.opening(morph);
            img.erode(morph);                   //erosion of opening
        }
    }

    img_with_count retVal = { img, 0 };
    retVal.img.clustering();
    retVal.object_count = retVal.img.countClusters();
    return retVal;
}

int main(int argc, char **argv) {
    assert(argc > 0);
    if (argc <= 2) {
        cout << "USAGE: skeleton <input file path>\nNumber inputs was " << argc << endl;
        return -1;
    }

    switch (argc){
        default:
        case 6:
            assert(atoi(argv[5]) == 1 || atoi(argv[5]) == 0);
            opening_gradient = (atoi(argv[5]) == 1);
        case 5:
            assert(atoi(argv[4]) == 1 || atoi(argv[4]) == 0);
            morph_gradient = (atoi(argv[4]) == 1);
        case 4:
            assert(atoi(argv[3]) == 1 || atoi(argv[3]) == 0);
            invertColors = (atoi(argv[3]) == 1);
        case 3:
            img_percent_threshold = atof(argv[2]);
        case 2:
            inputName = string(argv[1]);
        break;
        case 1:
            cout << "USAGE: ./sharpen <input file path> <CLUSTER_THRESHOLD> <USE_NEGATIVE> <NUMBER_ITERATIONS>" << endl;
            return -1;
        break;
    }

    Mat original_image  = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //original for display
    Mat greyscale_image = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //greyscaled for display
    Mat binary_image    = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //binary for display
    Mat preclust_image  = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //image before clustering
    Mat clust_image     = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //clustered for display
    Mat postclust_image = imread(inputName.c_str(), CV_LOAD_IMAGE_COLOR); //image after cluster trim
    Mat *image = &original_image;

    //Check that the images loaded
    if( NULL ==  original_image.data ||
        NULL == greyscale_image.data ||
        NULL ==    binary_image.data ||
        NULL ==  preclust_image.data ||
        NULL ==     clust_image.data ||
        NULL == postclust_image.data ) {
            cout << "ERROR: Could not load image data." << endl;
            return -1;
    }

    //Create the display window
    namedWindow("Image Object Counter");

    int COUNTS = 1;

    imageGrid original_img(original_image.rows, original_image.step / 3, &original_image.data[0]);
    imageGrid greyscale_img = original_img;
    if (invertColors) {
        greyscale_img.negative();
    } 
    imageGrid    binary_img = greyscale_img;
    imageGrid  preclust_img = greyscale_img;
    imageGrid     clust_img = greyscale_img;
    imageGrid postclust_img = original_img;
    greyscale_img.toGrey();
    binary_img.toBinary();    

    std::cout << "Processing with   Clustering Threshold:\t" << std::fixed << std::setprecision(6) << (img_percent_threshold * 100) << std::endl;
    std::cout << "Processing with         Image Negative:\t" << toString(invertColors) << std::endl;
    std::cout << "Processing with Morphological Gradient:\t" << toString(morph_gradient) << std::endl;
    std::cout << "Processing with       Opening Gradient:\t" << toString(opening_gradient) << std::endl;
    
    img_with_count processed_img = count_objects(preclust_img);
    clust_img = processed_img.img;
    postclust_img.cutWithImage(clust_img);

    std::cout << "There are " << processed_img.object_count << " in image.\n";

     original_img.commitImageGrid( original_image.data);
    greyscale_img.commitImageGrid(greyscale_image.data);
       binary_img.commitImageGrid(   binary_image.data);
     preclust_img.commitImageGrid( preclust_image.data);
        clust_img.commitImageGrid(    clust_image.data);
    postclust_img.commitImageGrid(postclust_image.data);

    unsigned state = 0;
    bool loop = true;
    while(loop) {
        imshow("Unix Sample Skeleton", *image);
        switch(cvWaitKey(15)) {
            case 27:  //Exit display loop if ESC is pressed
                loop = false;
            break;
            case 32:  //Swap image pointer if space is pressed
                switch(state % 6) {
                	case 0:
                		image = &original_image;
                		state++;
                		cout << "Original image.\n";
                	break;
                	case 1:
                		image = &greyscale_image;
                		state++;
                		cout << "GreyScale image.\n";
                	break;
                	case 2:
                		image = &binary_image;
                		state++;
                		cout << "Binary image.\n";
                	break;
                	case 3:
                		image = &preclust_image;
                		state++;
                		cout << "Image Before Structure Counting.\n";
                	break;
                	case 4:
                		image = &clust_image;
                		state++;
                		cout << "Structure Counted Image image.\n";
                	break;
                    case 5:
                        image = &postclust_image;
                        state++;
                        cout << "Original Image Masked with Counted Structures.\n";
                    break;
                	default:
                	break;
                }
            break;
            default:
            break;
        }
    }
}
