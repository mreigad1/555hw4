#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

#include "pixel.h"

#define MAX_CHAR (~((unsigned char) 0))

#define MAX(a,b) ((a > b)? a : b)
#define MIN(a,b) ((a < b)? a : b)
#define RANGE(LOW, X, HIGH) MAX(MIN(X, HIGH), 0)
#define MAX_COLOR (255.0)
#define CAP(X) RANGE(0, X, MAX_COLOR)
#define NUM_COLORS (3.0)
#define PI (3.141592653589793238462643383279502)
#define tPI (2 * PI)
#define EPS (2.71828182846)

extern double img_percent_threshold;
using namespace std;

enum BGR {
	RGB_B = 0,
	RGB_G = 1,
	RGB_R = 2
};

enum HSI {
	HSI_H = RGB_R,
	HSI_S = RGB_G,
	HSI_I = RGB_B
};

pixel::pixel(int R, int G, int B) {
	rgb[RGB_R] = R;
	rgb[RGB_G] = G;
	rgb[RGB_B] = B;
}

pixel::pixel() {
	rgb[RGB_R] = 0;
	rgb[RGB_G] = 0;
	rgb[RGB_B] = 0;
}

pixel pixel::lux() {
	pixelPrecision R = ( rgb[RGB_G] + rgb[RGB_B] ) / ( NUM_COLORS - 1 );
	pixelPrecision G = ( rgb[RGB_R] + rgb[RGB_B] ) / ( NUM_COLORS - 1 );
	pixelPrecision B = ( rgb[RGB_R] + rgb[RGB_G] ) / ( NUM_COLORS - 1 );
	pixel p_lux(R, G, B);
	return p_lux;
}

pixel pixel::unlux() {
	pixelPrecision R = ( rgb[RGB_G] + rgb[RGB_B] - rgb[RGB_R] );
	pixelPrecision G = ( rgb[RGB_R] + rgb[RGB_B] - rgb[RGB_G] );
	pixelPrecision B = ( rgb[RGB_R] + rgb[RGB_G] - rgb[RGB_B] );
	pixel p_lux(R, G, B);
	return p_lux;
}

pixel pixel::toGrey() {
	pixelPrecision R = ( rgb[RGB_R] + rgb[RGB_G] + rgb[RGB_B] ) / 3.0;
	pixelPrecision G = ( rgb[RGB_R] + rgb[RGB_G] + rgb[RGB_B] ) / 3.0;
	pixelPrecision B = ( rgb[RGB_R] + rgb[RGB_G] + rgb[RGB_B] ) / 3.0;
	pixel p(R, G, B);
	return p;
}

pixel pixel::negative() {
	pixelPrecision R = ( MAX_COLOR - rgb[RGB_R] );
	pixelPrecision G = ( MAX_COLOR - rgb[RGB_G] );
	pixelPrecision B = ( MAX_COLOR - rgb[RGB_B] );
	pixel p(R, G, B);
	return p;
}

pixel pixel::toBinary(pixelPrecision threshold) {
	pixel p = this->toGrey();
	pixelPrecision binVal = ((p.getAvgIntensity() >= threshold) ? MAX_COLOR : 0 );
	pixel retVal(binVal, binVal, binVal);
	return retVal;
}

pixelPrecision inBounds(pixelPrecision p) {
	while (p < 0.0) {
		p += tPI;
	}
	while (p >= tPI) {
		p -= tPI;
	}
	if (p < 0.0) {
		p += tPI;
	}
	//assert(p >= 0);
	//assert(p <= tPI);
	return p;
}

pixel pixel::RGB_toHSI() {
	pixel ret(0, 0, 0);
	pixelPrecision R = rgb[RGB_R];
	pixelPrecision G = rgb[RGB_G];
	pixelPrecision B = rgb[RGB_B];
	pixelPrecision M = MIN(MIN(R, G), B);
	pixelPrecision T = inBounds( acos((R - ((G + B) / 2.0)) / sqrt(R*R + G*G + B*B - R*G - R*B - G*B)) );
	//if (T >= 0) { std::cout << T << std::endl; }
	//assert(T < 0);
	//assert(T <= tPI);
	ret.rgb[HSI_H] = MAX_COLOR * ((G >= B) ? (T) : (tPI - T));
	ret.rgb[HSI_I] = (R + G + B) / NUM_COLORS;
	ret.rgb[HSI_S] = ((ret.rgb[HSI_I] > 0) ? (MAX_COLOR - (M / ret.rgb[HSI_I])) : (0));
	return ret;
}

pixel pixel::HSI_toRGB() {
	pixel ret(0,0,0);
	pixelPrecision H = inBounds(rgb[HSI_H] / MAX_COLOR);
	pixelPrecision S = rgb[HSI_S];
	pixelPrecision I = rgb[HSI_I];
	pixelPrecision R, G, B;
	pixelPrecision tmp;
	if (H == 0) {
		R = (I + 2 * S * I);
		G = (I - I * S);
		B = (I - I * S);
	} else if (H < tPI / 3) {
		tmp = cos(H) / cos(PI / 3 - H);
		R = I + I * S * tmp;
		G = I + I * S * (1 - tmp);
		B = I - I * S;
	} else if (H == tPI / 3) {
		R = I - I * S;
		G = I + 2 * I * S;
		B = I - I * S;
	} else if (H < 2 * tPI / 3) {
		tmp = cos(H - tPI / 3) / cos(PI - H);
		R = I - I * S;
		G = I + I * S * tmp;
		B = I + I * S * (1 - tmp);
	} else if (H == 2 * tPI / 3) {
		R = I - I * S;
		G = I - I * S;
		B = I + 2 * I * S;
	} else if (H > 2 * tPI / 3) {
		tmp = cos(H - 4 * PI / 3) / cos(5 * PI / 3 - H);
		R = I + I * S * (1 - tmp);
		G = I - I * S;
		B = I + I * S * tmp;
	} else {
		//std::cout << "ERROR!\n";
		//exit(0);
	}
	ret.rgb[RGB_R] = R;
	ret.rgb[RGB_G] = G;
	ret.rgb[RGB_B] = B;
	return ret;
}

pixelPrecision pixel::getAvgIntensity() {
	return toGrey().rgb[RGB_R];
}

bool pixel::operator<=(pixel& m) {
	pixel& p1 = *this;
	pixel& p2 = m;
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 <= i2;
}

bool pixel::operator<(pixel& m) {
	pixel& p1 = *this;
	pixel& p2 = m;
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 < i2;
}

bool pixel::operator>=(pixel& m) {
	pixel& p1 = *this;
	pixel& p2 = m;
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 >= i2;
}

bool pixel::operator>(pixel& m) {
	pixel& p1 = *this;
	pixel& p2 = m;
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 > i2;
}

bool pixel::operator==(pixel& m) {
	pixel& p1 = *this;
	pixel& p2 = m;
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 == i2;
}

bool operator<(const pixel& p1, const pixel& p2) {
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 < i2;
}

bool operator>(const pixel& p1, const pixel& p2) {
	unsigned long i1 = ((int(p1.rgb[RGB_R]) << 16) ^ (int(p1.rgb[RGB_G]) << 8) ^ int(p1.rgb[RGB_B]));
	unsigned long i2 = ((int(p2.rgb[RGB_R]) << 16) ^ (int(p2.rgb[RGB_G]) << 8) ^ int(p2.rgb[RGB_B]));
	return i1 > i2;
}

pixel pixel::operator+(const double& brightness){
	pixelPrecision R = rgb[RGB_R] + brightness;
	pixelPrecision G = rgb[RGB_G] + brightness;
	pixelPrecision B = rgb[RGB_B] + brightness;
	return pixel(R, G, B);
}

pixel pixel::operator+(const pixel& neighbor){
	pixelPrecision R = rgb[RGB_R] + neighbor.rgb[RGB_R];
	pixelPrecision G = rgb[RGB_G] + neighbor.rgb[RGB_G];
	pixelPrecision B = rgb[RGB_B] + neighbor.rgb[RGB_B];
	return pixel(R, G, B);
}

pixel pixel::operator-(const pixel& neighbor){
	pixelPrecision R = rgb[RGB_R] - neighbor.rgb[RGB_R];
	pixelPrecision G = rgb[RGB_G] - neighbor.rgb[RGB_G];
	pixelPrecision B = rgb[RGB_B] - neighbor.rgb[RGB_B];
	return pixel(R, G, B);
}

pixel pixel::operator/(const double& denom){
	assert(denom != 0.0);
	return (*this) * (1.0 / denom);
}

pixel pixel::operator*(const double& m){
	pixelPrecision R = rgb[RGB_R] * m;
	pixelPrecision G = rgb[RGB_G] * m;
	pixelPrecision B = rgb[RGB_B] * m;
	return pixel(R, G, B);
}

pixel pixel::operator*(const pixel& m) {
	pixelPrecision R = rgb[RGB_R] * rgb[RGB_R];
	pixelPrecision G = rgb[RGB_G] * rgb[RGB_G];
	pixelPrecision B = rgb[RGB_B] * rgb[RGB_B];
	return pixel(R, G, B);
}

pixel pixel::root() {
	pixelPrecision R = sqrt(rgb[RGB_R]);
	pixelPrecision G = sqrt(rgb[RGB_G]);
	pixelPrecision B = sqrt(rgb[RGB_B]);
	return pixel(R, G, B);
}

pixel pixel::mixP(double p) {
	pixelPrecision R = ((2 - p) * (rgb[RGB_B] + rgb[RGB_G] - rgb[RGB_R]) + p * rgb[RGB_R]) * 0.5;
	pixelPrecision G = ((2 - p) * (rgb[RGB_R] + rgb[RGB_B] - rgb[RGB_G]) + p * rgb[RGB_G]) * 0.5;
	pixelPrecision B = ((2 - p) * (rgb[RGB_R] + rgb[RGB_G] - rgb[RGB_B]) + p * rgb[RGB_B]) * 0.5;
	return pixel(R, G, B);
}

pixel_primitive pixel::toPixelPrimitive() {
	pixel_primitive p;
	p.rgb[RGB_B] = CAP(rgb[RGB_B]);
	p.rgb[RGB_G] = CAP(rgb[RGB_G]);
	p.rgb[RGB_R] = CAP(rgb[RGB_R]);
	return p;
}

mask::mask() {
	w = 0;
	maskVals = NULL;
}

mask::mask(unsigned int width, unsigned int listLength, double* initList, double coefficient) {
	assert((width * width) == listLength);
	w = width;
	double* curr = initList;
	maskVals = new double*[width];
	for (unsigned int i = 0; i < width; i++) {
		maskVals[i] = new double[width];
		for (unsigned int j = 0; j < width; j++) {
			maskVals[i][j] = (*(curr++)) * coefficient;
		}
	}
}

mask::~mask() {
	for (unsigned int i = 0; i < w; i++){
		delete[] maskVals[i];
	}
	delete[] maskVals;
}

mask& mask::operator=(const mask& other) {
	if (NULL != this->maskVals){
		for (unsigned int i = 0; i < w; i++) {
			delete[] this->maskVals[i];
			maskVals[i] = NULL;
		}
		delete[] maskVals;
		maskVals = NULL;
	}
	w = other.w;
	maskVals = new double*[w];
	for (unsigned int i = 0; i < w; i++) {
		maskVals[i] = new double[w];
		for (unsigned int j = 0; j < w; j++) {
			maskVals[i][j] = other.maskVals[i][j];
		}
	}
	return *this;
}

double mask::LOG(double x, double y, double sigma) {
	double sig2 = sigma * sigma;
	double sig4 = sig2 * sig2;
	double x2 = x * x;
	double y2 = y * y;
    double term1 = (-1.0 / (PI * sig4));
    double f_xy = ((x2 + y2) / (2.0 * sig2));
    double term2 = 1 - f_xy;
    double term3 = pow(EPS, -f_xy);
    return (term1 * term2 * term3);
}

mask mask::makeLOG(int width, double sigma) {
    //I don't wanna define this for even block size
    assert(width % 2 == 1);
    
    // Create the data array
    double* list = new double[width * width];

    double scaleFactor = MAX_COLOR;//-(MAX_COLOR) / LOG(0, 0, sigma);

    // Work out the "min and max" values. Log is centered around 0, 0
    // so, for a size 5 template (say) we want to get the values from
    // -2 to +2, ie: -2, -1, 0, +1, +2 and feed those into the formula.
    int min = -(width / 2);
    int max = (width / 2);

    // We also need a count to index into the data array...
    int xCount = 0;
    int yCount = 0;
    int lCount = 0;

    for(int x = min; x <= max; ++x) {
        for(int y = min; y <= max; ++y) {
            // Get the LoG value for this (x,y) pair
        	double ww = 1.0;
            double lx = (x / ww);
            double ly = (y / ww);
            list[lCount++] = LOG(lx, ly, sigma) * scaleFactor;
            ++yCount;
        }
        ++xCount;
    }

    mask retVal(width, width * width, list);
    delete[] list;
    return retVal;
}

pixel pixel_primitive::toPixel() {
	return pixel(MAX(rgb[RGB_R], 0), MAX(rgb[RGB_G], 0), MAX(rgb[RGB_B], 0));
}

imageGrid::imageGrid(unsigned height, unsigned width, unsigned char* old_data) {
	pixel_primitive* old = (pixel_primitive*) old_data;
	h = height;
	w = width;
	img = new pixel*[height];
	for (unsigned int i = 0; i < h; i++) {
		img[i] = new pixel[width];
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = (old++)->toPixel();
		}
	}
}

imageGrid::imageGrid() {
	img = NULL;
	h = 0;
	w = 0;
}

imageGrid::imageGrid(const imageGrid& other) {
	h = other.h;
	w = other.w;
	img = new pixel*[h];
	for (unsigned int i = 0; i < h; i++) {
		img[i] = new pixel[w];
		for (unsigned int j =0; j < w; j++) {
			img[i][j] = other.img[i][j];
		}
	}
}

imageGrid::imageGrid(imageGrid& other) {
	h = other.h;
	w = other.w;
	img = new pixel*[h];
	for (unsigned int i = 0; i < h; i++) {
		img[i] = new pixel[w];
		for (unsigned int j =0; j < w; j++) {
			img[i][j] = other.img[i][j];
		}
	}
}

imageGrid::~imageGrid() {
	if (NULL != img) {
		for (unsigned int i = 0; i < h; i++) {
			if (NULL != img[i]) {
				delete[] img[i];
			}
		}
		delete[] img;
	}
}

void imageGrid::RGB_toHSI() {
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].RGB_toHSI();
		}
	}
}
void imageGrid::HSI_toRGB() {
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].HSI_toRGB();
		}
	}
}

imageGrid& imageGrid::operator=(const imageGrid& other) {
	if (NULL != img){
		for (unsigned int i = 0; i < h; i++) {
			if (NULL != img[i]){
				delete[] img[i];
			}
			img[i] = NULL;
		}
		delete[] img;
		img = NULL;
	}
	h = other.h;
	w = other.w;
	img = new pixel*[h];
	for (unsigned int i = 0; i < h; i++) {
		img[i] = new pixel[w];
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = other.img[i][j];
		}
	}
	return *this;
}

void imageGrid::multiply(mask& _mask) {
	imageGrid buf;
	buf = *this;
	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.img[i][j] = multiplyPixel(i, j, _mask);
		}
	}
	*this = buf;
}

void imageGrid::dilate(mask& _mask) {
    imageGrid buf;
	buf = *this;
	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.img[i][j] = dilatePixel(i, j, _mask);
		}
	}
	*this = buf;
}

void imageGrid::dilateBinary(mask& _mask) {
	imageGrid buf;
	buf = *this;
	pixelPrecision threshold = getAvgIntensity();

	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.img[i][j] = dilatePixelBinary(i, j, _mask, threshold);
		}
	}
	*this = buf;
}

void imageGrid::erode(mask& _mask) {
	imageGrid buf;
	buf = *this;
	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.img[i][j] = erodePixel(i, j, _mask);
		}
	}
	*this = buf;
}

void imageGrid::opening(mask& _mask) {
	erode(_mask);
	dilate(_mask);
}

void imageGrid::closing(mask& _mask) {
	dilate(_mask);
	erode(_mask);
}

void imageGrid::morph_gradient(mask& _mask) {
	imageGrid erosion = *this;
	imageGrid dilation = *this;
	erosion.erode(_mask);
	dilation.dilate(_mask);
	dilation.subtract(erosion);
	*this = dilation;
}

void imageGrid::erodeBinary(mask& _mask) {
	imageGrid buf;
	buf = *this;
	pixelPrecision threshold = getAvgIntensity();

	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.img[i][j] = erodePixelBinary(i, j, _mask, threshold);
		}
	}
	*this = buf;
}

void imageGrid::negative() {
	//iterate over all pixels
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].negative();
		}
	}
}

void imageGrid::sobel() {

    double sob_x[] = { -1,  0,  1,
                       -2,  0,  2,
                       -1,  0,  1  };

    double sob_y[] = { -1, -2, -1,
                        0,  0,  0,
                        1,  2,  1  };

	mask sobelX(3, sizeof(sob_x) / sizeof(double), sob_x, 1 / 1.0);
    mask sobelY(3, sizeof(sob_y) / sizeof(double), sob_y, 1 / 1.0);

    imageGrid gx;
    imageGrid gy;
    gx = *this;
    gy = *this;

    gx.multiply(sobelX);
    gy.multiply(sobelY);

    for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = (gx.img[i][j] * gx.img[i][j] + gy.img[i][j] * gy.img[i][j]).root();
		}
	}
}

inline unsigned long rootHash(unsigned int i, unsigned int j, unsigned int h, unsigned int w){
	return (MAX_COLOR + (i * w) + j + 1);
}

inline unsigned char colorHash(double root, BGR color) {
	unsigned int k = root;
	unsigned char* arr = (unsigned char*)&k;
	k = (k << 4) | (arr[0] & 0x0F);
	unsigned char retVal = arr[(int)color];
	return retVal;
}

int find_pixel(std::vector<pixel>& roots, pixel& p) {
	for (int i = 0; i < roots.size(); i++) {
		if (roots[i] == p) {
			return i;
		}
	}
	return -1;
}

unsigned int imageGrid::countClusters() {
	std::vector<pixel> roots;
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			if (find_pixel(roots, img[i][j]) == -1 && img[i][j].getAvgIntensity() > 0) {
				roots.push_back(img[i][j]);
			}
			std::sort(roots.begin(), roots.end());
		}
	}
	return roots.size();
}

void imageGrid::clustering() {
	imageGrid buf;
	buf = *this;
	buf.toBinary();

	//iterate over all pixels, and shove large roots into all tree-clusters
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			buf.clusterPixel(i, j, rootHash(i, j, h, w));
		}
	}

	//iterate over all pixels, and shove large roots into all tree-clusters
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			double root = buf.img[i][j].getAvgIntensity();
			if (root > 0) {
				buf.img[i][j] = pixel(
					colorHash(root, RGB_R),
					colorHash(root, RGB_G),
					colorHash(root, RGB_B)
				);
			}
		}
	}

	*this = buf;
}

void imageGrid::clusterPixel(unsigned int y, unsigned int x, unsigned int root) {

	assert(y < h && x < w);
	assert(root > MAX_COLOR);
	unsigned int intensity = img[y][x].getAvgIntensity();
	if (intensity <= 0 || (intensity < root && intensity != MAX_COLOR)) {
		return;
	}
	
	std::vector<v2> cluster;
	cluster.push_back(v2(y, x));

	for (unsigned int i = 0; i < cluster.size(); i++) {
		y = cluster[i].coord[0];
		x = cluster[i].coord[1];
		int my_I = img[y][x].getAvgIntensity();
		img[y][x] = pixel(root, root, root);
		unsigned int y_l = ((y == 0)?(0):(y - 1));
		unsigned int x_l = ((x == 0)?(0):(x - 1));
		unsigned int y_h = MIN(y + 1, h - 1);
		unsigned int x_h = MIN(x + 1, w - 1);
		assert(y_l >= 0 && x_l >= 0 && x_h < w && y_h < h);
		for (unsigned int n_y = y_l; n_y <= y_h; n_y++) {
			for (unsigned int n_x = x_l; n_x <= x_h; n_x++) {		//for each neighbor
				if (n_x == x || n_y == y) {
					int I = img[n_y][n_x].getAvgIntensity();	//downcast intensity
					unsigned int I_delta = MAX(I, my_I) - MIN(I, my_I);
					if (I > root || I == MAX_COLOR) {				//if neighbor larger or just alive
						cluster.push_back(v2(n_y, n_x));
						img[n_y][n_x] = pixel(root, root, root);
					}
				}
			}
		}
	}

	//objects must make up at least X% of image or be discarded (noise)
	if (cluster.size() < (h * w * img_percent_threshold)) {
		for (unsigned int i = 0; i < cluster.size(); i++) {
			y = cluster[i].coord[0];
			x = cluster[i].coord[1];
			img[y][x] = pixel(0, 0, 0);
		}
	}
}

pixel imageGrid::multiplyPixel(unsigned int y, unsigned int x, mask& _mask) {
	pixel retVal(0, 0, 0);
	int off = _mask.w / 2;
	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				retVal = retVal + (img[y_prime][x_prime] * _mask.maskVals[y_off + off][x_off + off]);
			} else {
				retVal = retVal + (img[y][x] * _mask.maskVals[y_off + off][x_off + off]);
			}
		}
	}
	return retVal;
}

pixel imageGrid::dilatePixel(unsigned int y, unsigned int x, mask& _mask) {
	pixel retVal(0, 0, 0);
	int off = _mask.w / 2;
	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				pixel tempPix = img[y_prime][x_prime].toGrey() + _mask.maskVals[y_off + off][x_off + off];
				retVal = ((retVal > tempPix) ? retVal : tempPix);
			} 
		}
	}
	return retVal;
}

pixel imageGrid::dilatePixelBinary(unsigned int y, unsigned int x, mask& _mask, pixelPrecision threshold) {
	pixel retVal(0, 0, 0);
	int off = _mask.w / 2;
	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				int enable = ((_mask.maskVals[y_off + off][x_off + off] != 0.0) ? 1 : 0);
				pixel tempPix = img[y_prime][x_prime].toBinary(threshold) * enable;
				retVal = ((tempPix.getAvgIntensity() > 0) ? tempPix : retVal);
			} 
		}
	}
	return retVal;
}

pixel imageGrid::erodePixel(unsigned int y, unsigned int x, mask& _mask) {
	pixel retVal(MAX_COLOR, MAX_COLOR, MAX_COLOR);
	int off = _mask.w / 2;
	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				pixel tempPix = img[y_prime][x_prime].toGrey() + (-_mask.maskVals[y_off + off][x_off + off]);
				retVal = ((retVal < tempPix) ? retVal : tempPix);
			} 
		}
	}
	return retVal;
}

pixel imageGrid::erodePixelBinary(unsigned int y, unsigned int x, mask& _mask, pixelPrecision threshold) {
	pixel retVal(MAX_COLOR, MAX_COLOR, MAX_COLOR);
	int off = _mask.w / 2;
	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				int enable = ((_mask.maskVals[y_off + off][x_off + off] != 0.0) ? 1 : 0);
				if (enable) {
					pixel tempPix = img[y_prime][x_prime].toBinary(threshold);
					retVal = ((tempPix.getAvgIntensity() > 0 && retVal.getAvgIntensity() > 0) ? retVal : pixel(0,0,0));
				}
			} 
		}
	}
	return retVal;
}

void imageGrid::medianFilter(unsigned int width) {
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = medianFiltered(i, j, width);
		}
	}
}

void imageGrid::cutWithImage(imageGrid& logical_mask) {
	assert(h == logical_mask.h && w == logical_mask.w);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (logical_mask.img[i][j].getAvgIntensity() > 0) {

			} else {
				img[i][j] = pixel(0,0,0);
			}
		}
	}
}

pixel imageGrid::medianFiltered(unsigned int y, unsigned int x, unsigned int width) {
	pixel retVal(0, 0, 0);
	int off = width / 2;

	std::vector<pixel> neighborhood;

	//iterate over all valid pixels in neighborhood
	for (int y_off = -off; y_off <= off; y_off++) {
		int y_prime = y + y_off;
		for (int x_off = -off; x_off <= off; x_off++) {
			int x_prime = x + x_off;
			if (x_prime >= 0 && x_prime < w && y_prime >= 0 && y_prime < h) {
				neighborhood.push_back(img[y_prime][x_prime]);
			}
		}
	}

	std::sort(neighborhood.begin(), neighborhood.end());
	retVal = neighborhood[neighborhood.size() / 2];

	return retVal;
}

void imageGrid::mixWith(imageGrid& other, double other_ratio) {
	int min_w = MIN(other.w, w);
	int min_h = MIN(other.h, h);

	for (unsigned int i = 0; i < min_h; i++) {
		for (unsigned int j = 0; j < min_w; j++) {
			img[i][j] = (img[i][j] * (1 - other_ratio) + other.img[i][j] * other_ratio);
		}
	}
}

void imageGrid::geometricMix(imageGrid& other) {
	int min_w = MIN(other.w, w);
	int min_h = MIN(other.h, h);

	for (unsigned int i = 0; i < min_h; i++) {
		for (unsigned int j = 0; j < min_w; j++) {
			img[i][j] = (img[i][j] * other.img[i][j]).root();
		}
	}
}

void imageGrid::mixP(double p) {
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].mixP(p);
		}
	}
}

void imageGrid::lux() {
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].lux();
		}
	}
}

void imageGrid::unlux() {
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].unlux();
		}
	}
}

void imageGrid::toGrey() {
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].toGrey();
		}
	}
}

void imageGrid::toBinary() {
	pixelPrecision threshold = getAvgIntensity();
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j].toBinary(threshold);
		}
	}
}

void imageGrid::subtract(imageGrid& other) {
	assert(h == other.h && w == other.w);
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = MAX(img[i][j] - other.img[i][j], pixel(0,0,0));
		}
	}
}

void imageGrid::add(imageGrid& other) {
	assert(h == other.h && w == other.w);
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			img[i][j] = img[i][j] - other.img[i][j];
		}
	}
}

pixelPrecision imageGrid::getAvgIntensity() {
	pixelPrecision threshold = 0;
	for (unsigned int i = 0; i < h; i++){
		for (unsigned int j = 0; j < w; j++) {
			threshold += img[i][j].getAvgIntensity();
		}
	}
	threshold /= (h * w);
	return threshold;
}

void imageGrid::commitImageGrid(unsigned char * old_data) {
	pixel_primitive* old = (pixel_primitive*) old_data;
	for (unsigned int i = 0; i < h; i++) {
		for (unsigned int j = 0; j < w; j++) {
			(*old++) = img[i][j].toPixelPrimitive();
		}
	}
}

