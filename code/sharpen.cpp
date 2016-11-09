#include <unistd.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "pixel.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if(argc != 2) {
        cout << "USAGE: skeleton <input file path>" << endl;
        return -1;
    }

    //Load two copies of the image. One to leave as the original, and one to be modified.
    //Done for display purposes only
    //Use CV_LOAD_IMAGE_GRAYSCALE for greyscale images
    Mat original_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat unsharp_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat sobel_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat log9_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat log11_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    //Create a pointer so that we can quickly toggle between which image is being displayed
    Mat *image = &original_image;

    //Check that the images loaded
    if(!original_image.data || !unsharp_image.data) {
        cout << "ERROR: Could not load image data." << endl;
        return -1;
    }

    //Create the display window
    namedWindow("Unix Sample Skeleton");

    double blur_list[] = {  0, -1,  0,
                           -1,  5, -1,
                            0, -1,  0  };

    double dilate_list[] = {  1, 0, 0, 0, 1,
                              0, 1, 0, 1, 0,
                              0, 0, 1, 0, 0,
                              0, 1, 0, 1, 0,
                              1, 0, 0, 0, 1  };

    mask   blur(3, sizeof(blur_list) / sizeof(double), blur_list, 1 / 1.0);
    mask dilate(5, sizeof(dilate_list) / sizeof(double), dilate_list, 1 / 1.0);

    int COUNTS = 1;

    imageGrid unsharp_img(unsharp_image.rows, unsharp_image.step / 3, &unsharp_image.data[0]);
    unsharp_img.multiply(blur);
    unsharp_img.commitImageGrid(&unsharp_image.data[0]);

    imageGrid sobel_img(sobel_image.rows, sobel_image.step / 3, &sobel_image.data[0]);
    sobel_img.sobel();
    sobel_img.commitImageGrid(&sobel_image.data[0]);

    int num_it = 3;
    double mix_ratio = -0.05;

	imageGrid unHSI_img(log9_image.rows, log9_image.step / 3, &log9_image.data[0]);
    unHSI_img.toGrey();
    unHSI_img.commitImageGrid(&log9_image.data[0]);

	imageGrid HSI_img(log11_image.rows, log11_image.step / 3, &log11_image.data[0]);
    HSI_img.toGrey();
    HSI_img.erode(dilate);
    HSI_img.commitImageGrid(&log11_image.data[0]);

    unsigned state = 0;

    //Display loop
    bool loop = true;
    while(loop) {
        imshow("Unix Sample Skeleton", *image);
    
        switch(cvWaitKey(15)) {
            case 27:  //Exit display loop if ESC is pressed
                loop = false;
            break;
            case 32:  //Swap image pointer if space is pressed
                if(image == &original_image) {
                  image = &unsharp_image;
                } else {
                  image = &original_image;
                }
                switch(state % 5) {
                	case 0:
                		image = &original_image;
                		state++;
                		cout << "Original image.\n";
                	break;
                	case 1:
                		image = &unsharp_image;
                		state++;
                		cout << "Unsharp image.\n";
                	break;
                	case 2:
                		image = &sobel_image;
                		state++;
                		cout << "luxed image.\n";
                	break;
                	case 3:
                		image = &log9_image;
                		state++;
                		cout << "binary image.\n";
                	break;
                	case 4:
                		image = &log11_image;
                		state--;
                		cout << "dilated image.\n";
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
