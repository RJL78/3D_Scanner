//
//  ViewController.m
//  3D_Scanner
//
//  Created by Raphael Laporte on 4/11/17.
//  Copyright © 2017 Raphael_Laporte. All rights reserved.
//

#import  <AVFoundation/AVFoundation.h>
#import  <AVFoundation/AVCaptureVideoPreviewLayer.h>
#import  "ViewController.h"
#include "myfit.h"

#ifdef __cplusplus
#include <stdlib.h>
#import  <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "armadillo"
#endif


@interface ViewController () {

    int64 curr_time_;

    UIImageView *cameraView_;
    UIImageView *renderedView_;
    UITextView *fpsView_;

   
    std::vector<cv::KeyPoint> last_frame_keypoints;
    std::vector<cv::KeyPoint> curr_frame_keypoints;
    cv::Mat last_frame_descriptors;
    cv::Mat curr_frame_descriptors;
    
    arma::fmat::fixed<3,3> currentHomography;
    arma::fmat::fixed<3,3> stepHomography;
    bool firstFrame;
    
    cv::SurfFeatureDetector *SURFdetector_;
    cv::SurfDescriptorExtractor *SURFextractor_;
    
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    
    

    
}
@end

@implementation ViewController

@synthesize videoCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    
    firstFrame = true;
    int minHessian = 400;
    SURFextractor_ = new cv::SurfDescriptorExtractor();
    SURFdetector_  = new cv::SurfFeatureDetector(minHessian);
    matcher        = cv::BFMatcher::BFMatcher(cv::NORM_L2,true);

    currentHomography << 1 << 0 << 0 << arma::endr
                      << 0 << 1 << 0 << arma::endr
                      << 0 << 0 << 1;
    
   float cam_width = 720; float cam_height = 1280;
    
    // Take into account size of camera input
    int view_width = self.view.frame.size.width;
    int view_height = (int)(cam_height*self.view.frame.size.width/cam_width);
    int offset = (self.view.frame.size.height - view_height)/2;
    
    cameraView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, offset, view_width, view_height)];
    
    [self.view addSubview:cameraView_]; // Add the view
    
    // Initialize the video camera
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView_];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30; // Set the frame rate
    self.videoCamera.grayscaleMode = YES; // Get grayscale
    self.videoCamera.rotateVideo = YES; // Rotate video so everything looks correct
    
    // Choose these depending on the camera input chosen
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1280x720;
    
    // Finally add the FPS text to the view
    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
    [fpsView_ setOpaque:false]; // Set to be Opaque
    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
    [self.view addSubview:fpsView_];
    
    [videoCamera start];
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}



-  (void) processImage:(cv::Mat &) cvImage
{


    // STEP 1: Process current frame into gray
    cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 1.2, 1.2); // Apply Gaussian blur
 
    // STEP 2: Extract descriptors from current frame
    SURFdetector_->detect(gray, curr_frame_keypoints);
    SURFextractor_->compute(gray, curr_frame_keypoints, curr_frame_descriptors);
    
    if (!firstFrame){
    
        // STEP 3 : Calculate homography between last frame and current frame
        matcher.match(last_frame_descriptors, curr_frame_descriptors, matches);
        currentHomography = my_ransac(matches, last_frame_keypoints, curr_frame_keypoints);
        
        // STEP 4: Project Current Frame onto 3D model
    
    
        // STEP 5: Render 3D Model
        
        
        
    }
    firstFrame = false;
    
    // STEP 5: Update Class Variables for next iteration
    last_frame_descriptors = curr_frame_descriptors.clone();
    last_frame_keypoints = curr_frame_keypoints;
    
 
    cv::Scalar RED(0,0,255);
    for (size_t i =0; i<matches.size() ; i++){
        cv::Point point = curr_frame_keypoints[matches[i].trainIdx].pt;
        cv::circle(cvImage,point,2,RED);
    }
    
    // STEP 6: Display FPS 
    int64 next_time = getTickCount(); // Get the next time stamp
    float fps = (float)getTickFrequency()/(next_time - curr_time_); // Estimate the fps
    curr_time_ = next_time; // Update the time
    NSString *fps_NSStr = [NSString stringWithFormat:@"FPS = %2.2f",fps]
    
    
    
    
    
    
    
}







// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

@end


