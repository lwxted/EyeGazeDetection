//
//  ViewController.m
//  EyeExtraction
//
//  Created by Ted Li on 4/16/17.
//  Copyright © 2017 Ted Li. All rights reserved.
//

#import "ViewController.h"
#import "UIImage+Crop.h"

#import <GoogleMobileVision/GoogleMobileVision.h>

@interface ViewController ()

@property (nonatomic, strong) GMVDetector *faceDetector;

@end

@implementation ViewController

- (NSArray<UIImage *> *) extractEyeRegion:(NSString *)fromPicturePath {
    UIImage *image = [UIImage imageWithContentsOfFile:fromPicturePath];
    NSArray<GMVFaceFeature *> *faces = [self.faceDetector featuresInImage:image options:nil];
    NSMutableArray<UIImage *> *eyeRegions = [NSMutableArray array];
    for (GMVFaceFeature *face in faces) {
        CGRect faceRect = face.bounds;
        float eyeWidth = faceRect.size.width * 0.28;
        float eyeHeight = faceRect.size.height * 0.20;
        if (face.hasLeftEyePosition) {
            CGPoint leftEyePos = face.leftEyePosition;
            CGRect leftEyeRect = CGRectMake(leftEyePos.x - eyeWidth / 2.0, leftEyePos.y - eyeHeight / 2.0, eyeWidth, eyeHeight);
            UIImage *leftEye = [image crop:leftEyeRect];
            [eyeRegions addObject:leftEye];
        }
        
        if (face.hasRightEyePosition) {
            CGPoint rightEyePos = face.rightEyePosition;
            CGRect rightEyeRect = CGRectMake(rightEyePos.x - eyeWidth / 2.0, rightEyePos.y - eyeHeight / 2.0, eyeWidth, eyeHeight);
            UIImage *rightEye = [image crop:rightEyeRect];
            [eyeRegions addObject:rightEye];
        }
    }
    return eyeRegions;
}

- (void)convertAllImagesOfPath:(NSString *)fromPath toPath:(NSString *)toPath {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:toPath]) {
        [fileManager createDirectoryAtPath:toPath withIntermediateDirectories:YES attributes:nil error:nil];
    }
    NSArray<NSString *> *directoryContents = [fileManager contentsOfDirectoryAtPath:fromPath error:nil];
    for (NSString *fileName in directoryContents) {
        if ([fileName.pathExtension isEqualToString:@"jpg"]) {
            NSString *fromPicturePath = [fromPath stringByAppendingPathComponent:fileName];
            NSArray<UIImage *> *eyeRegionImages = [self extractEyeRegion:fromPicturePath];
            if (eyeRegionImages.count < 2) {
                NSLog(@"%@/%@: %d eyes found", toPath, fileName, (int) eyeRegionImages.count);
            }
            for (int i = 0; i < eyeRegionImages.count; ++i) {
                UIImage *eyeImage = eyeRegionImages[i];
                NSString *imageFileName = [[fileName lastPathComponent] stringByDeletingPathExtension];
                NSString *eyeFileName = [imageFileName stringByAppendingFormat:@"_%d.jpg", i];
                NSString *eyeFilePath = [toPath stringByAppendingPathComponent:eyeFileName];
                [UIImageJPEGRepresentation(eyeImage, 1.0) writeToFile:eyeFilePath atomically:YES];
            }
        }
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSDictionary *options = @{
                              GMVDetectorFaceMinSize : @(0.3),
                              GMVDetectorFaceTrackingEnabled : @(YES),
                              GMVDetectorFaceLandmarkType : @(GMVDetectorFaceLandmarkAll),
                              GMVDetectorFaceMode : @(GMVDetectorFaceAccurateMode)
                              };
    self.faceDetector = [GMVDetector detectorOfType:GMVDetectorTypeFace options:options];
    
    NSArray<NSString *> *folderNames = @[@"00.Centre",
                                         @"01.UpRight",
                                         @"02.UpLeft",
                                         @"03.Right",
                                         @"04.Left",
                                         @"05.DownRight",
                                         @"06.DownLeft"
                                         ];
    
    for (NSString *folderName in folderNames) {
        NSString *fromPath = [NSString stringWithFormat:@"/Users/lwxted/Documents/ml/eye_gaze/data.nosync/Eye_chimeraToPublish/%@/", folderName];
        NSString *toPath = [NSString stringWithFormat:@"/Users/lwxted/Documents/ml/eye_gaze/data_eye.nosync/%@/", folderName];
        [self convertAllImagesOfPath:fromPath toPath:toPath];
    }
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
