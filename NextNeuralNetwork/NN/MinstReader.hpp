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

class MinstReader{
    uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }
public:
    void read_mnist(const char* image_filename, const char* label_filename){
        // Open files
        std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
        std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
        
        // Read the magic and the meta data
        uint32_t magic;
        uint32_t num_items;
        uint32_t num_labels;
        uint32_t rows;
        uint32_t cols;
        
        image_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2051){
            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
            return;
        }
        
        label_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2049){
            std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
            return;
        }
        
        image_file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = swap_endian(num_items);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = swap_endian(num_labels);
        if(num_items != num_labels){
            std::cout<<"image file nums should equal to label num"<<std::endl;
            return;
        }
        
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        rows = swap_endian(rows);
        image_file.read(reinterpret_cast<char*>(&cols), 4);
        cols = swap_endian(cols);
        
        std::cout<<"image and label num is: "<<num_items<<std::endl;
        std::cout<<"image rows: "<<rows<<", cols: "<<cols<<std::endl;
        
        char label;
        char* pixels = new char[rows * cols];
        
        for (int item_id = 0; item_id < num_items; ++item_id) {
            // read image pixel
            image_file.read(pixels, rows * cols);
            // read label
            label_file.read(&label, 1);
            
            std::string sLabel = std::to_string(int(label));
            std::cout<<"lable is: "<<sLabel<<std::endl;
        }
        
        delete[] pixels;
    }
};

#endif /* MinstReader_hpp */
