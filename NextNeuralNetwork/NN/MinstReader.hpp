//
//  MinstReader.hpp
//  NextNeuralNetwork
//
//  Created by Liang,Zhiyuan(GIS)2 on 27/9/18.
//  Copyright © 2018年 Ogreaxe. All rights reserved.
//

#ifndef MinstReader_hpp
#define MinstReader_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "NaiveNeuralNet.hpp"

class MinstReader{
    NaiveNeuralNet<char, float> * neuralNet = nullptr;
    
    uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }
public:
    void read_mnist(const char* image_filename, const char* label_filename,const char* test_image_filename, const char* test_label_filename){
        // Open files
        std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
        std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
        
        uint32_t num_items;
        uint32_t num_labels;
        uint32_t num_testItems;
        uint32_t num_testLabels;
        uint32_t rows;
        uint32_t cols;
        
        if(!read_header(image_file, label_file, rows, cols, num_items, num_labels)){
            return;
        }
        
        std::ifstream test_image_file(test_image_filename, std::ios::in | std::ios::binary);
        std::ifstream test_label_file(test_label_filename, std::ios::in | std::ios::binary);
        
        if(!read_header(test_image_file, test_label_file, rows, cols, num_testItems, num_testLabels)){
            return;
        }
        
        num_testItems = 100;
//        image_file.read(reinterpret_cast<char*>(&magic), 4);
//        magic = swap_endian(magic);
//        if(magic != 2051){
//            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
//            return;
//        }
//
//        label_file.read(reinterpret_cast<char*>(&magic), 4);
//        magic = swap_endian(magic);
//        if(magic != 2049){
//            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
//            return;
//        }
//
//        image_file.read(reinterpret_cast<char*>(&num_items), 4);
//        num_items = swap_endian(num_items);
//        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
//        num_labels = swap_endian(num_labels);
//        if(num_items != num_labels){
//            std::cout<<"image file nums should equal to label num"<<std::endl;
//            return;
//        }
//
//        image_file.read(reinterpret_cast<char*>(&rows), 4);
//        rows = swap_endian(rows);
//        image_file.read(reinterpret_cast<char*>(&cols), 4);
//        cols = swap_endian(cols);
//
//        std::cout<<"image and label num is: "<<num_items<<std::endl;
//        std::cout<<"image rows: "<<rows<<", cols: "<<cols<<std::endl;
        
        NaiveNeuralNet<char, float> * nn = new NaiveNeuralNet<char, float>();
        nn->setDataStructure(rows * cols, 10);
        if (this->neuralNet) {
            delete this->neuralNet;
        }
        this->neuralNet = nn;
        
        //test data
        char* pixels_test = new char[rows * cols * num_testItems];
        
        float* labels_test = new float[10 * num_testItems];
        createTestData(test_image_file, test_label_file, pixels_test, labels_test, rows, cols, num_testItems);
        nn->setTestData(pixels_test, labels_test, num_testItems);
        
        //train data
        int batchsize = 100;

        char* pixels_batch = new char[rows * cols * batchsize];
        
        float* labels_batch = new float[10 * batchsize];
        
        char* labels_batch_original = new char[ batchsize];
        
        for (int item_id = 0; item_id < num_items; item_id += batchsize) {
            // read image pixel
            image_file.read(pixels_batch, rows * cols * batchsize);
            // read label
            label_file.read(labels_batch_original, batchsize);
            
            for (int i=0; i<batchsize; i++) {
                for (int shift = 0; shift < 10; shift ++) {
                    labels_batch[i*10 + shift] =0; ;
                }
                
                labels_batch[i*10 + labels_batch_original[i]] = 1;
            }
            
            nn->setTrainData(pixels_batch, labels_batch, batchsize);
            nn->setTestData(pixels_batch, labels_batch, batchsize);
            for(int i=0;i<200;i++){
                if(nn->train(batchsize)){
                    printf("train success");
                    break;
                }
                nn->realtest();
            }
        }
        
        delete[] pixels_batch;
        delete[] labels_batch;
        delete[] labels_batch_original;
//        char label;
//        char* pixels = new char[rows * cols];
//
//        for (int item_id = 0; item_id < num_items; ++item_id) {
//            // read image pixel
//            image_file.read(pixels, rows * cols);
//            // read label
//            label_file.read(&label, 1);
//
//            std::string sLabel = std::to_string(int(label));
//            std::cout<<"lable is: "<<sLabel<<std::endl;
//        }
//
//        delete[] pixels;
    }
    
    bool read_header(std::ifstream & image_file, std::ifstream & label_file, uint32_t & rows,
                    uint32_t & cols, uint32_t & num_items, uint32_t & num_labels){
        // Read the magic and the meta data
        uint32_t magic;
        
        image_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2051){
            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
            return false;
        }
        
        label_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2049){
            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
            return false;
        }
        
        image_file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = swap_endian(num_items);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = swap_endian(num_labels);
        if(num_items != num_labels){
            std::cout<<"image file nums should equal to label num"<<std::endl;
            return false;
        }
        
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        rows = swap_endian(rows);
        image_file.read(reinterpret_cast<char*>(&cols), 4);
        cols = swap_endian(cols);
        
        std::cout<<"image and label num is: "<<num_items<<std::endl;
        std::cout<<"image rows: "<<rows<<", cols: "<<cols<<std::endl;
        
        return true;
    }
    
    void createTestData(std::ifstream & test_image_file, std::ifstream & test_label_file,char* pixels, float* labels, uint32_t rows,
                        uint32_t cols, uint32_t num_items){
        char labels_original;
        
        for (int item_id = 0; item_id < num_items; item_id += 1) {
            // read image pixel
            test_image_file.read(pixels, rows * cols);
            // read label
            test_label_file.read(&labels_original, 1);
            
            
            for (int shift = 0; shift < 10; shift ++) {
                labels[item_id*10 + shift] =0; ;
            }
            labels[item_id*10 + labels_original] = 1;
        }
    }
};

#endif /* MinstReader_hpp */
