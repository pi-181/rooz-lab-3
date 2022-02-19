#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


class ColorDetector {

private:
    // minimum acceptable distance
    int maxDist;

    // target color
    cv::Vec3b target;

    // image containing color converted image
    cv::Mat converted;

    // 0 - bgr
    // 1 - lab
    // 2 - hsv
    int labHsv;

    // image containing resulting binary map
    cv::Mat result;

public:

    // empty constructor
    // default parameter initialization here
    ColorDetector() : maxDist(100), target(0,0,0), labHsv(0) {}

    // extra constructor for Lab color space example
    explicit ColorDetector(int labHsv) : maxDist(100), target(0,0,0), labHsv(labHsv) {}

    // full constructor
    ColorDetector(uchar blue, uchar green, uchar red, int mxDist=100, int labHsv=0): maxDist(mxDist), labHsv(labHsv) {
        // target color
        setTargetColor(blue, green, red);
    }

    // Computes the distance from target color.
    [[nodiscard]]
    int getDistanceToTargetColor(const cv::Vec3b& color) const {
        return getColorDistance(color, target);
    }

    // Computes the city-block distance between two colors.
    [[nodiscard]]
    int getColorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) const {
        return abs(color1[0]-color2[0])+
               abs(color1[1]-color2[1])+
               abs(color1[2]-color2[2]);
    }

    // Processes the image. Returns a 1-channel binary image.
    cv::Mat process(const cv::Mat &image);

    cv::Mat operator()(const cv::Mat &image) {
        cv::Mat input;

        if (labHsv == 1) { // Lab conversion
            cv::cvtColor(image, input, cv::COLOR_BGR2Lab);
        } else if (labHsv == 2) {
            cv::cvtColor(image, input, cv::COLOR_BGR2HSV);
        } else {
            input = image;
        }

        cv::Mat output;
        // compute absolute difference with target color
        cv::absdiff(input,cv::Scalar(target),output);
        // split the channels into 3 images
        std::vector<cv::Mat> images;
        cv::split(output,images);
        // add the 3 channels (saturation might occurs here)
        output= images[0]+images[1]+images[2];
        // apply threshold
        cv::threshold(output,
                      output,
                      maxDist, // threshold ( < 256)
                      255,
                      cv::THRESH_BINARY_INV);

        return output;
    }

    // Getters and setters
    void setColorDistanceThreshold(int distance) {
        if (distance<0)
            distance=0;
        maxDist= distance;
    }

    [[nodiscard]]
    int getColorDistanceThreshold() const {
        return maxDist;
    }

    void setTargetColor(uchar blue, uchar green, uchar red) {
        target = cv::Vec3b(blue, green, red);

        if (labHsv != 0) {
            // Temporary 1-pixel image
            cv::Mat tmp(1, 1, CV_8UC3);
            tmp.at<cv::Vec3b>(0, 0) = cv::Vec3b(blue, green, red);

            // Converting the target to Lab color space
            cv::cvtColor(tmp, tmp, labHsv == 1 ? cv::COLOR_BGR2Lab : cv::COLOR_BGR2HSV);

            target = tmp.at<cv::Vec3b>(0, 0);
        }
    }

    // Sets the color to be detected
    void setTargetColor(const cv::Vec3b& color) {
        target = color;
    }

    // Gets the color to be detected
    [[nodiscard]]
    cv::Vec3b getTargetColor() const {
        return target;
    }
};


cv::Mat ColorDetector::process(const cv::Mat &image) {
    result.create(image.size(),CV_8U);

    if (labHsv == 1) {
        cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    } else if (labHsv == 2) {
        cv::cvtColor(image, converted, cv::COLOR_BGR2HSV);
    }

    cv::Mat_<cv::Vec3b>::const_iterator it= image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::const_iterator itend= image.end<cv::Vec3b>();
    cv::Mat_<uchar>::iterator itout= result.begin<uchar>();

    if (labHsv != 0) {
        it = converted.begin<cv::Vec3b>();
        itend = converted.end<cv::Vec3b>();
    }

    // for each pixel
    for ( ; it!= itend; ++it, ++itout) {
        // compute distance from target color
        if (getDistanceToTargetColor(*it)<maxDist) {
            *itout= 255;
        } else {
            *itout= 0;
        }
    }

    return result;
}

void detectHScolor(const cv::Mat& image, // input image
                   double minHue, double maxHue, // Hue interval
                   double minSat, double maxSat, // saturation interval
                   cv::Mat& mask) { // output mask
    // convert into HSV space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    // split the 3 channels into 3 images
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // Hue masking
    cv::Mat mask1; // under maxHue
    cv::threshold(channels[0], mask1, maxHue, 255,cv::THRESH_BINARY_INV);
    cv::Mat mask2; // over minHue
    cv::threshold(channels[0], mask2, minHue, 255,cv::THRESH_BINARY);
    cv::Mat hueMask; // hue mask

    if (minHue < maxHue)
        hueMask = mask1 & mask2;
    else // if interval crosses the zero-degree axis
        hueMask = mask1 | mask2;

    // under maxSat
    cv::threshold(channels[1], mask1, maxSat, 255,cv::THRESH_BINARY_INV);
    // over minSat
    cv::threshold(channels[1], mask2, minSat, 255,cv::THRESH_BINARY);

    cv::Mat satMask; // saturation mask
    satMask = mask1 & mask2;

    // combined mask
    mask = hueMask & satMask;
}

const cv::String picPath = R"(pics\1.jpg)";
const cv::String photoPath = R"(pics\2.jpg)";

int main() {
    ColorDetector cdetect;

    // 1. Завантаження зображень
    Mat image = imread(picPath);
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    // 2. Виконати програмну реалізацію алгоритму для розпізнавання заданого кольору на зображенні.
    // В якості міри приналежності кольору заданому використовувати різницю значень кольорів пікселів
    cdetect.setTargetColor(0xe7, 0xbb, 0x9e);

    namedWindow("Using absdiff Image", WINDOW_AUTOSIZE);
    imshow("Using absdiff Image", cdetect.process(image));

    // 3. Виконати розпізнавання кольору на зображенні, використовуючи функцію floodFill.
    Mat floodFillImage = image.clone();
    cv::floodFill(floodFillImage,            // input/ouput image
                  cv::Point(10, 10),         // seed point
                  cv::Scalar(255, 255, 255),  // repainted color
                  nullptr,  // bounding rectangle of the repainted pixel set
                  cv::Scalar(35, 35, 35),     // low and high difference threshold
                  cv::Scalar(35, 35, 35),     // most of the time will be identical
                  cv::FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

    cv::namedWindow("Flood Fill result");
    cv::imshow("Flood Fill result", floodFillImage);

    // 4. Виконати сегментацію зображення за допомогою алгоритму GrabCut.

    const Rect2i foregroundRect = Rect(0, 250, 366, 399);

    Mat marked = image.clone();
    rectangle(marked, foregroundRect, Scalar(0, 0, 255));
    imshow("Selected area", marked);

    cv::Mat result; // segmentation (4 possible values)
    cv::Mat bgModel,fgModel;

    cv::grabCut(image, // input image
                result, // segmentation result
                foregroundRect,// rectangle containing foreground
                bgModel, fgModel, // models
                5, // number of iterations
                cv::GC_INIT_WITH_RECT); // use rectangle

    // Get the pixels marked as likely foreground
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(image.size(),CV_8UC3,cv::Scalar(255,255,255));
    image.copyTo(foreground, result);

    cv::namedWindow("GrabCut result");
    cv::imshow("GrabCut result", foreground);

    // 5. Виконати розпізнавання кольору для зображення у колірних системах LAB та HSV.

    Mat labImg;
    cvtColor(image, labImg, COLOR_BGR2Lab);

    ColorDetector cldetect(1);
    cldetect.setTargetColor(0xe7, 0xbb, 0x9e);

    namedWindow("Lab detect Image", WINDOW_AUTOSIZE);
    imshow("Lab detect Image", cldetect.process(labImg));

    Mat hsvImg;
    cvtColor(image, hsvImg, COLOR_BGR2HSV);

    ColorDetector chdetect(2);
    chdetect.setTargetColor(0xe7, 0xbb, 0x9e);

    namedWindow("HSV detect Image", WINDOW_AUTOSIZE);
    imshow("HSV detect Image", chdetect.process(labImg));

    // 6.	Встановити постійну яскравість усім пікселям зображення в системі HSV.
    // Відобразіть результат та його гістограму.

    std::vector<cv::Mat> channels;
    cv::split(hsvImg,channels);

    // Канал значення = 255 для всіх пікселів
    channels[2]= 255;

    Mat hsvFixedValueImage;
    cv::merge(channels, hsvFixedValueImage);

    cv::Mat fixedValueImage;
    cv::cvtColor(hsvFixedValueImage, fixedValueImage, COLOR_HSV2BGR);

    namedWindow("Fixed Value Image", WINDOW_AUTOSIZE);
    imshow("Fixed Value Image", fixedValueImage);

    // 7. Виконати розпізнавання тону шкіри.

    Mat photo = imread(photoPath);
    resize(photo, photo, Size(photo.cols / 6, photo.rows / 6));

    imshow("7. Source Image", photo);

    cv::Mat mask;
    detectHScolor(photo,
                  160, 10, // hue from 320 degrees to 20 degrees
                  25, 166, // saturation from ~0.1 to 0.65
                  mask);

    // show masked image
    cv::Mat detected(photo.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    photo.copyTo(detected, mask);

    namedWindow("7. Skin Image", WINDOW_AUTOSIZE);
    imshow("7. Skin Image", detected);

    waitKey(0);
    waitKey(0);

    return 0;
}
