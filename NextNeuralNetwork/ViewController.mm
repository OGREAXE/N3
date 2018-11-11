//
//  ViewController.m
//  NextNeuralNetwork
//
//  Created by 梁志远 on 2018/6/18.
//  Copyright © 2018 Ogreaxe. All rights reserved.
//

#import "ViewController.h"
#include "MinstReader.hpp"
#include <string>
using namespace std;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        [self startTraining];
    });
}

-(void)startTraining{
    NSString *docpath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
    docpath = [docpath stringByAppendingString:@"/minst/"];
    string base_dir = docpath.UTF8String;
    string img_path = base_dir + "train-images-idx3-ubyte";
    string label_path = base_dir + "train-labels-idx1-ubyte";
    string test_img_path = base_dir + "t10k-images-idx3-ubyte";
    string test_label_path = base_dir + "t10k-labels-idx1-ubyte";
    
    MinstReader minstReader;
    
    minstReader.read_mnist(img_path.c_str(), label_path.c_str(),test_img_path.c_str(), test_label_path.c_str());
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
