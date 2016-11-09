class pixel;
class pixel_primitive;
class imageGrid;
class mask;

typedef double pixelPrecision;

//pixel stores high precision pixel data during
//mathematical operations to avoid overflows
class pixel {
	public:
		pixel();
		pixel(int R, int G, int B);
		pixel operator+(const pixel& neighbor);
		pixel operator+(const double& brightness);
		pixel operator-(const pixel& neighbor);
		pixel operator/(const double& denom);
		pixel operator*(const double& m);
		pixel operator*(const pixel& m);
		bool operator<=(pixel& m);
		bool operator<(pixel& m);
		bool operator>=(pixel& m);
		bool operator>(pixel& m);
		pixel lux();
		pixel unlux();
		pixel root();
		pixel mixP(double p);
		pixel_primitive toPixelPrimitive();
		pixel RGB_toHSI();
		pixel HSI_toRGB();
		pixel toGrey();
		pixelPrecision getAvgIntensity();
		pixel toBinary();
	private:
		static const int PIX_ARR_SIZE = 3;
		pixelPrecision rgb[PIX_ARR_SIZE];
};

//pixel primitive is a type with size and alignment matching native pixel
//implementation and provides a mechanism for casting to higher precision type
class pixel_primitive {
	public:
		pixel toPixel();
		friend pixel_primitive pixel::toPixelPrimitive();
	private:
		static const int pp_size = 3;
		unsigned char rgb[pp_size];
};

//imageGrid stores a copy of an imageGrid to be operated upon and allows copying
//back to imageGrid buffer following operations
class imageGrid {
	public:
		imageGrid();
		imageGrid(unsigned height, unsigned width, unsigned char* old_data);
		~imageGrid();
		imageGrid& operator=(const imageGrid& other);
		void multiply(mask& _mask);
		void dilate(mask& _mask);
		void dilateBinary(mask& _mask);
		void erode(mask& _mask);
		void erodeBinary(mask& _mask);
		void sobel();
		pixel multiplyPixel(unsigned int y, unsigned int x, mask& _mask);
		pixel dilatePixel(unsigned int y, unsigned int x, mask& _mask);
		pixel dilatePixelBinary(unsigned int y, unsigned int x, mask& _mask);
		pixel erodePixel(unsigned int y, unsigned int x, mask& _mask);
		pixel erodePixelBinary(unsigned int y, unsigned int x, mask& _mask);
		void RGB_toHSI();
		void HSI_toRGB();
		void lux();
		void unlux();
		void toGrey();
		void toBinary();
		void mixP(double p);
		void mixWith(imageGrid& other, double other_ratio = 0.5);
		void geometricMix(imageGrid& other);
		void commitImageGrid(unsigned char* old_data);
	private:
		unsigned int h;
		unsigned int w;
		pixel** img;
};

//mask stores data for square masks filters
class mask {
	public:
		mask();
		mask(unsigned int width, unsigned int listLength, double* initList, double coefficient = 1.0);
		friend pixel imageGrid::multiplyPixel(unsigned int y, unsigned int x, mask& _mask);
		friend pixel imageGrid::dilatePixel(unsigned int y, unsigned int x, mask& _mask);
		friend pixel imageGrid::dilatePixelBinary(unsigned int y, unsigned int x, mask& _mask);
		friend pixel imageGrid::erodePixel(unsigned int y, unsigned int x, mask& _mask);
		friend pixel imageGrid::erodePixelBinary(unsigned int y, unsigned int x, mask& _mask);
		static double LOG(double x, double y, double sigma);
		static mask makeLOG(int width, double sigma);
		~mask();
		mask& operator=(const mask& other);
	private:
		unsigned int w;
		double** maskVals;
};

