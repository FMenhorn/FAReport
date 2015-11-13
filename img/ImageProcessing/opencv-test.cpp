#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

void help() {
	cout
			<< "\nThis program demonstrates line finding with the Hough transform.\n"
					"Usage:\n"
					"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

Mat houghLines(Mat img, int angleSteps, int distanceSteps) {

	// TODO: nicht optimal, es muss nur der halbe Winkelbereich durchgegangen werden
	// und es wäre besser für die Distanzen, wenn der Ursprung im Zentrum sitzt
	int width = img.cols;
	int height = img.rows;
	float angleStepwidth = 2 * M_PI / angleSteps;
	float maxDist = sqrt(width * width + height * height);
	int halfDistanceSteps = distanceSteps / 2;
	float stepsPerDist = halfDistanceSteps / maxDist;

	Mat result = Mat::zeros(distanceSteps, angleSteps, CV_32S);

	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			if (img.at<uint8_t>(y, x) > 0) {
				float angle = 0.0;
				for (int i = 0; i < angleSteps; ++i) {
					// TODO: Rundung hier ist nicht korrekt und optimalerweise muss das min und das max auch nicht hier hin
					int32_t distanceIndex = halfDistanceSteps
							+ static_cast<int32_t>(stepsPerDist
									* (x * sin(angle) + y * cos(angle)) + 0.5);
					++result.at<int32_t>(
							std::max(0,
									std::min(distanceSteps - 1, distanceIndex)),
							i);
					angle += angleStepwidth;
				}
			}
		}
	}

	return result;
}

template<typename T> T max(Mat img) {
	T maxValue = std::numeric_limits<T>::min();
	for (int x = 0; x < img.cols; ++x) {
		for (int y = 0; y < img.rows; ++y) {
			T val = img.at<T>(y, x);
			if (val > maxValue) {
				maxValue = val;
			}
		}
	}

	return maxValue;
}

template<typename T> T min(Mat img) {
	T minValue = std::numeric_limits<T>::max();
	for (int x = 0; x < img.cols; ++x) {
		for (int y = 0; y < img.rows; ++y) {
			T val = img.at<T>(y, x);
			if (val < minValue) {
				minValue = val;
			}
		}
	}

	return minValue;
}

template<typename From, typename To> Mat normalize(Mat img, bool useMin = true,
		To newMin = std::numeric_limits<To>::min(), To newMax =
				std::numeric_limits<To>::max()) {
	From minVal = useMin ? min<From>(img) : From(0);
	From maxVal = max<From>(img);
	From fromDist = maxVal - minVal;

	if (fromDist <= 0) {
		fromDist = 1;
	}

	Mat dest(img.rows, img.cols, DataType<To>::type);

	for (int x = 0; x < img.cols; ++x) {
		for (int y = 0; y < img.rows; ++y) {
			auto val = newMin
					+ (img.at<From>(y, x) - minVal) * (newMax - newMin) / fromDist;
			dest.at<To>(y, x) = static_cast<To>(std::max(
					static_cast<decltype(val)>(newMin),
					std::min(static_cast<decltype(val)>(newMax), val)));
		}
	}

	return dest;
}

template<typename From, typename To> Mat convert(Mat img) {
	Mat dest(img.rows, img.cols, DataType<To>::type);

	for (int x = 0; x < img.cols; ++x) {
		for (int y = 0; y < img.rows; ++y) {
			dest.at<To>(y, x) = static_cast<To>(img.at<From>(y, x));
		}
	}

	return dest;
}

template<typename In1, typename In2, typename Out, typename Func> Mat applyFunction(
		Mat first, Mat second, Func func) {
	if (first.cols != second.cols || first.rows != second.rows) {
		std::cout << "No matching size!\n";
		return Mat();
	}

	Mat result(first.rows, first.cols, DataType<Out>::type);

	for (int x = 0; x < first.cols; ++x) {
		for (int y = 0; y < first.rows; ++y) {
			result.at<Out>(y, x) = func(first.at<In1>(y, x),
					second.at<In2>(y, x));
		}
	}

	return result;
}

int main(int argc, char** argv) {
	std::string filename = argc >= 2 ? argv[1] : "img.png";
	filename = "../data/" + filename;

	Mat src = imread(filename, 0);
	if (src.empty()) {
		help();
		cout << "can not open " << filename << endl;
		return -1;
	}

	Mat hough = houghLines(src, 200, 200);

	Mat dest = normalize<int32_t, uint8_t>(hough);

	Mat averageFilter = (1.0 / 25.0) * Mat::ones(5, 5, CV_32F);
	float gaussArray[3][3] = { { 1.0, 2.0, 1.0 }, { 2.0, 4.0, 2.0 }, { 1.0, 2.0,
			1.0 } };
	float sobelXArray[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	float sobelYArray[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
	float laplaceArray[3][3] = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };
	// Mat gaussianFilter = (1.0 / 16.0) * Mat(3, 3, CV_32F, gaussArray);
	Mat gaussianFilter = getGaussianKernel(7, 1.0, CV_32F);
	Mat sobelXFilter = Mat(3, 3, CV_32F, sobelXArray);
	Mat sobelYFilter = Mat(3, 3, CV_32F, sobelYArray);
	Mat laplaceFilter = Mat(3, 3, CV_32F, laplaceArray);

	Mat avgDest(src.rows, src.cols, CV_32F);
	Mat gaussDest(src.rows, src.cols, CV_32F);
	Mat sobelXDest(src.rows, src.cols, CV_32F);
	Mat sobelYDest(src.rows, src.cols, CV_32F);
	Mat laplaceDest(src.rows, src.cols, CV_32F);

	filter2D(src, avgDest, CV_32F, averageFilter, Point(-1, -1), 0,
			BORDER_REFLECT);
	filter2D(src, gaussDest, CV_32F, gaussianFilter, Point(-1, -1), 0,
			BORDER_REFLECT);
	filter2D(gaussDest, sobelXDest, CV_32F, sobelXFilter, Point(-1, -1), 0,
			BORDER_REFLECT);
	filter2D(gaussDest, sobelYDest, CV_32F, sobelYFilter, Point(-1, -1), 0,
			BORDER_REFLECT);
	filter2D(gaussDest, laplaceDest, CV_32F, laplaceFilter, Point(-1, -1), 0,
			BORDER_REFLECT);

	Mat sobelAbs = applyFunction<float, float, float>(sobelXDest, sobelYDest,
			[](float x, float y) -> float {return sqrt(x*x + y*y);});
	Mat sobelDir = applyFunction<float, float, float>(sobelXDest, sobelYDest,
			[](float x, float y) -> float {return atan2(y, x);});

	Mat gauss = convert<float, uint8_t>(gaussDest);
	Mat cannyDest(gauss.rows, gauss.cols, CV_8U);

	Canny(gauss, cannyDest, 40, 120);

	imwrite("../data/average.png", convert<float, uint8_t>(avgDest));
	imwrite("../data/gauss.png", convert<float, uint8_t>(gaussDest));
	imwrite("../data/sobelx.png", normalize<float, uint8_t>(sobelXDest));
	imwrite("../data/sobely.png", normalize<float, uint8_t>(sobelYDest));
	imwrite("../data/laplace.png", normalize<float, uint8_t>(laplaceDest));
	imwrite("../data/sobelabs.png", normalize<float, uint8_t>(sobelAbs));
	imwrite("../data/sobeldir.png", normalize<float, uint8_t>(sobelDir));
	imwrite("../data/original.png", src);
	imwrite("../data/canny.png", cannyDest);

	imwrite("../data/hough.png", dest);

	return 0;
}

