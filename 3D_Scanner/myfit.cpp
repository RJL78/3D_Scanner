//
//  myfit.cpp
//  Estimate_Homography
//
//  Created by Simon Lucey on 9/21/15.
//  Copyright (c) 2015 CMU_16432. All rights reserved.
//

#include "myfit.h"
#include "math.h"
#include "stdlib.h"
#include <set>

#include <opencv2/opencv.hpp>


// Use the Armadillo namespace
using namespace arma;

//-----------------------------------------------------------------
// Function to return the affine warp between 3D points on a plane
//
// <in>
// X = concatenated matrix of 2D projected points in the image (2xN)
// W = concatenated matrix of 3D points on the plane (3XN)
//
// <out>
// A = 2x3 matrix of affine parameters
fmat myfit_affine(fmat &X, fmat &W) {

    // Fill in the answer here.....
    fmat A; return A;
}

    
// Function to return the affine warp between 3D points on a plane
//
// <in>
// X = concatenated matrix of 2D projected points in the image (2xN)
// W = concatenated matrix of 3D points on the plane (3XN)
//
// <out>
// H = 3x3 homography matrix
    
fmat myfit_homography(fmat &X, fmat &W) {
    
    


    arma::fmat A;
    A.set_size(2*size(X).n_cols, 9);
    for (int i=0; i<size(X).n_cols; i++){
          // Build two rows of matrix A
        arma::fmat::fixed<1,9> temp;
        temp  << -W(0,i) << -W(1,i) << -1 << 0 << 0 << 0 << W(0,i)*X(0,i) << W(1,i)*X(0,i) << X(0,i);
        A.row(2*i) = temp;
        temp  << 0 << 0 << 0 << -W(0,i) << -W(1,i) << -1 << W(0,i)*X(1,i) << W(1,i)*X(1,i) << X(1,i);
        A.row(2*i+1) = temp;
        
    }

    // Solve x for Ax = 0
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd(U, s, V, A);
    arma::fmat x = V.col(size(V).n_cols-1);
    
 
    x.reshape(3,3);
    x = x.t();

    
    if (x(2,2) < 0){
      x.transform( [](float val) { return val *= -1; } );
    }
    
    return x;
}

int random(int max)
{
    int output = (rand()*max / RAND_MAX);
 
    return output;
}

arma::fmat my_ransac(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kpnts1,
                                  std::vector<cv::KeyPoint> kpnts2)
{
    int subSampleSize = 4;
    std::vector<cv::DMatch> filteredMatches;
    
    for (int i=0 ; i<matches.size(); i++){
        
        if (matches[i].distance< 0.70){
          filteredMatches.push_back(matches[i]);
        }
    }
    
    std::cout << filteredMatches.size() << endl;


    std::vector<cv::DMatch>  mostInliers;
    float maxInlierDist = 4;
    int leastOutliers = INT_MAX;
    
    int numTries = 0;
    int maxTries = 25000;
    
    while (numTries<maxTries){
        
        std::vector<cv::DMatch>  inliers;
        
        // Sample keyPoints in original View
        std::set<int> randomIndices;
        
        while (randomIndices.size()<subSampleSize){
            randomIndices.insert(rand()%filteredMatches.size() );
        }
        
        // Build X and Y
        fmat X;
        fmat Y;
        X.set_size(2,subSampleSize);
        Y.set_size(3,subSampleSize);
        
        int i = 0;
        for (int randomIndex: randomIndices){
            
            cv::Point2f origViewPt = kpnts1[filteredMatches[randomIndex].queryIdx].pt;
            cv::Point2f newViewPt  = kpnts2[filteredMatches[randomIndex].trainIdx].pt;
            
            fmat temp;
            temp << origViewPt.x << endr << origViewPt.y << endr << 1;
            Y.col(i) = temp;
            
            temp << newViewPt.x << endr << newViewPt.y;
            X.col(i) = temp;
            i+=1;
        
        }
        
        // Find Homograpy
        fmat homography = myfit_homography(X, Y);
        
        // Count Outliers
        int numOutliers = 0;
        for ( cv::DMatch match: filteredMatches){
            cv::Point2f origViewPt = kpnts1[match.queryIdx].pt;
            cv::Point2f newViewPt  = kpnts2[match.trainIdx].pt;
            
            arma::fmat::fixed<3,1> toProject;
            toProject << origViewPt.x << endr << origViewPt.y << endr << 1;
            
            arma::fmat::fixed<3,1> projResult =  homography*toProject;
       
            
            if ( (projResult(0,0)/projResult(2,0)-newViewPt.x)*(projResult(0,0)/projResult(2,0)-newViewPt.x) +
                 (projResult(1,0)/projResult(2,0)-newViewPt.y)*(projResult(1,0)/projResult(2,0)-newViewPt.y) >
                 maxInlierDist *maxInlierDist){
                 numOutliers += 1;
            }
            else
            {
                inliers.push_back(match);
            }
        }
        
        
        
        if (leastOutliers > numOutliers){
            leastOutliers = numOutliers;
            mostInliers   = inliers;
            std::cout <<  numTries << std::endl;
            std::cout <<  leastOutliers << std::endl;
        }
    
        numTries += 1;
        if (numTries%1000 ==0){
            std::cout <<  numTries << std::endl;
        }

    }
    
    fmat X,Y;
    X.set_size(2,mostInliers.size());
    Y.set_size(3,mostInliers.size());
    

    
    for ( int i = 0; i<mostInliers.size(); i++){
        cv::Point2f origViewPt = kpnts1[mostInliers[i].queryIdx].pt;
        cv::Point2f newViewPt  = kpnts2[mostInliers[i].trainIdx].pt;
        fmat temp;
        temp << origViewPt.x << endr << origViewPt.y << endr << 1;
        Y.col(i) = temp;
        
        temp << newViewPt.x << endr << newViewPt.y;
        X.col(i) = temp;
    }
    
    return( myfit_homography(X, Y) );
}


// Function to project points using the affine transform
//
// <in>
// W = concatenated matrix of 3D points on the plane (3XN)
// H = 3x3 homography matrix
//
// <out>
// X = concatenated matrix of 2D projected points in the image (2xN)
fmat myproj_homography(fmat &W, fmat &H) {
    
    arma::fmat X;
    X.set_size(2, size(W).n_cols);
    for (int i=0; i<size(W).n_cols; i++){
        arma::fmat::fixed<3,1> point = W.col(i);
        point(2,0) = 1;
        point = H*point;
        X(0,i) = point(0,0)/point(2,0);
        X(1,i) = point(1,0)/point(2,0);
    }
    
    std::cout<< X << std::endl;
    
    return X;
}

//-----------------------------------------------------------------
// Function to project points using the affine transform
//
// <in>
// W = concatenated matrix of 3D points on the plane (3XN)
// A = 2x3 matrix of affine parameters
//
// <out>
// X = concatenated matrix of 2D projected points in the image (2xN)
fmat myproj_affine(fmat &W, fmat &A) {
    
    
    
    // Fill in the answer here.....
    fmat X; return X;
}
