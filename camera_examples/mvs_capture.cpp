#include <MvCameraControl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check(int result, const char* operation) {
    if (result != MV_OK) {
        throw std::runtime_error(
            std::string(operation) + " failed: 0x" +
            std::to_string(static_cast<unsigned int>(result)));
    }
}

std::string fixedString(const unsigned char* data, std::size_t size) {
    const auto* begin = reinterpret_cast<const char*>(data);
    return std::string(begin, std::find(begin, begin + size, '\0'));
}

}  // namespace

int main(int argc, char** argv) {
    void* handle = nullptr;
    bool initialized = false;
    bool opened = false;
    bool grabbing = false;
    bool leased = false;
    MV_FRAME_OUT frame{};
    int exitCode = 0;

    try {
        check(MV_CC_Initialize(), "MV_CC_Initialize");
        initialized = true;
        MV_CC_DEVICE_INFO_LIST devices{};
        check(MV_CC_EnumDevices(MV_USB_DEVICE, &devices), "MV_CC_EnumDevices");

        MV_CC_DEVICE_INFO* selected = nullptr;
        for (unsigned int i = 0; i < devices.nDeviceNum; ++i) {
            auto* device = devices.pDeviceInfo[i];
            if (!device || device->nTLayerType != MV_USB_DEVICE) continue;
            const auto& usb = device->SpecialInfo.stUsb3VInfo;
            const auto serial = fixedString(usb.chSerialNumber, sizeof(usb.chSerialNumber));
            const auto model = fixedString(usb.chModelName, sizeof(usb.chModelName));
            std::cout << "model=" << model << " serial=" << serial << '\n';
            if (argc > 1 && serial == argv[1]) selected = device;
        }
        if (argc < 2) {
            std::cout << "Usage: mvs_capture SERIAL [BayerRG8|BGR8] [FRAMES]\n";
            throw std::runtime_error("serial argument required");
        }
        if (!selected) throw std::runtime_error("serial number not found");

        const std::string format = argc > 2 ? argv[2] : "BayerRG8";
        unsigned int pixelFormat = PixelType_Gvsp_BayerRG8;
        if (format == "BGR8") pixelFormat = PixelType_Gvsp_BGR8_Packed;
        else if (format != "BayerRG8")
            throw std::runtime_error("format must be BayerRG8 or BGR8");
        const int frameLimit = argc > 3 ? std::stoi(argv[3]) : 10;

        check(MV_CC_CreateHandle(&handle, selected), "MV_CC_CreateHandle");
        check(MV_CC_OpenDevice(handle), "MV_CC_OpenDevice");
        opened = true;
        check(MV_CC_SetEnumValueByString(handle, "AcquisitionMode", "Continuous"),
              "AcquisitionMode");
        check(MV_CC_SetEnumValueByString(handle, "TriggerMode", "Off"), "TriggerMode");
        check(MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off"), "ExposureAuto");
        check(MV_CC_SetEnumValueByString(handle, "ExposureMode", "Timed"), "ExposureMode");
        check(MV_CC_SetEnumValueByString(handle, "GainAuto", "Off"), "GainAuto");
        check(MV_CC_SetEnumValue(handle, "PixelFormat", pixelFormat), "PixelFormat");

        MVCC_FLOATVALUE gain{};
        check(MV_CC_GetFloatValue(handle, "Gain", &gain), "Get Gain range");
        check(MV_CC_SetFloatValue(handle, "ExposureTime", 1000.0f), "ExposureTime");
        check(MV_CC_SetFloatValue(handle, "Gain", gain.fMin), "Gain");
        check(MV_CC_StartGrabbing(handle), "MV_CC_StartGrabbing");
        grabbing = true;

        cv::Mat converted;
        cv::Mat saved;
        for (int count = 0; count < frameLimit;) {
            frame = {};
            const int result = MV_CC_GetImageBuffer(handle, &frame, 1000);
            if (static_cast<unsigned int>(result) == MV_E_NODATA) {
                std::cerr << "timeout waiting for a frame\n";
                continue;
            }
            check(result, "MV_CC_GetImageBuffer");
            leased = true;

            const auto& info = frame.stFrameInfo;
            const int width = info.nWidth;
            const int height = info.nHeight;
            if (!frame.pBufAddr || width <= 0 || height <= 0)
                throw std::runtime_error("invalid image dimensions or pointer");
            const auto type = info.enPixelType;
            const bool directBgr = type == PixelType_Gvsp_BGR8_Packed;
            const bool bayer = type == PixelType_Gvsp_BayerRG8;
            if (!directBgr && !bayer)
                throw std::runtime_error("unsupported pixel format");
            const std::size_t rowBytes =
                static_cast<std::size_t>(width) * (directBgr ? 3 : 1);
            if (rowBytes * static_cast<std::size_t>(height) != info.nFrameLen)
                throw std::runtime_error("padded frames are not handled by this example");

            cv::Mat image;
            if (directBgr) {
                image = cv::Mat(height, width, CV_8UC3, frame.pBufAddr, rowBytes);
            } else {
                converted.create(height, width, CV_8UC3);
                MV_CC_PIXEL_CONVERT_PARAM_EX params{};
                params.nWidth = width;
                params.nHeight = height;
                params.enSrcPixelType = type;
                params.pSrcData = frame.pBufAddr;
                params.nSrcDataLen = info.nFrameLen;
                params.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
                params.pDstBuffer = converted.data;
                params.nDstBufferSize =
                    static_cast<unsigned int>(converted.total() * converted.elemSize());
                check(MV_CC_ConvertPixelTypeEx(handle, &params), "ConvertPixelTypeEx");
                image = converted;
            }
            // The copy is owned by OpenCV and remains valid after the SDK lease
            // is returned. The borrowed ``image`` itself must not be retained.
            if (count == 0) saved = image.clone();

            check(MV_CC_FreeImageBuffer(handle, &frame), "MV_CC_FreeImageBuffer");
            frame = {};
            leased = false;
            image.release();
            ++count;
        }
        if (!saved.empty() && !cv::imwrite("mvs_first_frame.png", saved))
            throw std::runtime_error("failed to write mvs_first_frame.png");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        exitCode = 1;
    }

    if (leased) {
        if (MV_CC_FreeImageBuffer(handle, &frame) != MV_OK) exitCode = 1;
        std::cerr << "[清理] FreeImageBuffer\n";
    }
    if (grabbing) {
        if (MV_CC_StopGrabbing(handle) != MV_OK) exitCode = 1;
        std::cerr << "[清理] StopGrabbing\n";
    }
    if (opened) {
        if (MV_CC_CloseDevice(handle) != MV_OK) exitCode = 1;
        std::cerr << "[清理] CloseDevice\n";
    }
    if (handle) {
        if (MV_CC_DestroyHandle(handle) != MV_OK) exitCode = 1;
        std::cerr << "[清理] DestroyHandle\n";
    }
    if (initialized && MV_CC_Finalize() != MV_OK) exitCode = 1;
    return exitCode;
}
