#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

int main() {
    // 1. A cv::Mat can borrow another buffer. Mat copies are shallow by default.
    std::vector<unsigned char> buffer(2 * 16, 0);
    cv::Mat borrowed(2, 4, CV_8UC3, buffer.data(), 16);
    cv::Mat alias = borrowed;
    cv::Mat owned = borrowed.clone();

    borrowed.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    std::cout << "shared_pointer=" << (borrowed.data == alias.data)
              << " cloned_pointer=" << (borrowed.data != owned.data)
              << " step=" << borrowed.step[0]
              << " continuous=" << borrowed.isContinuous() << '\n';
    std::cout << "alias_pixel=" << alias.at<cv::Vec3b>(0, 0)
              << " owned_pixel=" << owned.at<cv::Vec3b>(0, 0) << '\n';

    // 2. A Bayer frame is single-channel CFA data, not an OpenCV BGR image.
    cv::Mat bayer(8, 8, CV_8UC1);
    for (int y = 0; y < bayer.rows; ++y) {
        for (int x = 0; x < bayer.cols; ++x) {
            bayer.at<unsigned char>(y, x) =
                (y % 2 == 0)
                    ? (x % 2 == 0 ? 200 : 100)
                    : (x % 2 == 0 ? 100 : 20);
        }
    }
    cv::Mat bgr;
    cv::cvtColor(bayer, bgr, cv::COLOR_BayerRGGB2BGR);
    std::cout << "BayerRG8_to_BGR_pixel=" << bgr.at<cv::Vec3b>(3, 3) << '\n';

    // 3. For MVS: clone before MV_CC_FreeImageBuffer. After freeing the SDK
    // buffer, the borrowed Mat and its shallow aliases are no longer valid.
    std::cout << "MVS rule: one successful GetImageBuffer <-> one FreeImageBuffer; "
                 "clone borrowed frame data before freeing.\n";
    return 0;
}
