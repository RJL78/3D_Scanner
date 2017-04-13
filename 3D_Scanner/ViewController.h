//
//  ViewController.h
//  3D_Scanner
//
//  Created by Raphael Laporte on 4/11/17.
//  Copyright © 2017 Raphael_Laporte. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <opencv2/highgui/ios.h>

// Slightly changed things here to employ the CvVideoCameraDelegate
@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    CvVideoCamera *videoCamera; // OpenCV class for accessing the camera
}
// Declare internal property of videoCamera
@property (nonatomic, retain) CvVideoCamera *videoCamera;

@end
