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
#include <string>
#include "NaiveNeuralNet.hpp"
#include <time.h>

using namespace std;

class MinstReader{
    NaiveNeuralNet<char, double> * neuralNet = nullptr;
    
    uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }
    
private:
    int m_batchSize = 1;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t num_testItems;
    uint32_t num_testLabels;
    uint32_t rows;
    uint32_t cols;
    
    string image_filename;
    string label_filename;
    string test_image_filename;
    string test_label_filename;
    
    string timeStr(){
        char tmp[64];
        time_t t = time(NULL);
        tm *_tm = localtime(&t);
        int year  = _tm->tm_year+1900;
        int month = _tm->tm_mon+1;
        int date  = _tm->tm_mday;
        int hh = _tm->tm_hour;
        int mm = _tm->tm_min;
        int ss = _tm->tm_sec;
        sprintf(tmp,"%02d:%02d:%02d>",hh,mm,ss);
        return string(tmp);
    }
public:
    void read_mnist(const char* image_filename, const char* label_filename,const char* test_image_filename, const char* test_label_filename){
        
        this->image_filename = image_filename;
        this->label_filename = label_filename;
        this->test_image_filename = test_image_filename;
        this->test_label_filename = test_label_filename;
        
        testMiniBatch();
    }
    
    void testMiniBatch(){
        // Open files
        std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
        std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
        
        if(!read_header(image_file, label_file, rows, cols, num_items, num_labels)){
            printf("read error");
            return;
        }
        
        std::ifstream test_image_file(test_image_filename, std::ios::in | std::ios::binary);
        std::ifstream test_label_file(test_label_filename, std::ios::in | std::ios::binary);
        
        if(!read_header(test_image_file, test_label_file, rows, cols, num_testItems, num_testLabels)){
            return;
        }
        
        NaiveNeuralNet<char, double> * nn = new NaiveNeuralNet<char, double>();
        nn->setDataStructure(rows * cols, 10);
        if (this->neuralNet) {
            delete this->neuralNet;
        }
        this->neuralNet = nn;
        
        //test data
        char* pixels_test = new char[rows * cols * num_testItems];
        
        double* labels_test = new double[10 * num_testItems];
        createTestData(test_image_file, test_label_file, pixels_test, labels_test, rows, cols, num_testItems);
        nn->setTestData(pixels_test, labels_test, num_testItems);
        
        char* pixels_batch = nullptr;
        double* labels_batch= nullptr;  //0~9
        char* labels_batch_original= nullptr ;
        
        int pass = 0;
        
        char* pixels_all = new char[rows * cols * num_items];
        
        double* labels_all = new double[10 * num_items]; //0~9
        
        char* labels_all_original = new char[ num_items];
        
        //            nn->lookAnInput(pixels_test, labels_test, 5000);
        
        image_file.read(pixels_all, rows * cols * num_items);
        // read label
        label_file.read(labels_all_original, num_items);
        
        
        printf("%s read finish",timeStr().c_str());
        for (int i=0; i<num_items; i++) {
            for (int shift = 0; shift < 10; shift ++) {
                labels_all[i*10 + shift] =0; ;
            }
            
            labels_all[i*10 + labels_all_original[i]] = 1;
        }
        
        while (1) {
            //train data
            int batchsize = m_batchSize;
            
            int step = batchsize;
//            int step = 1;
            for (int item_id = 0; item_id < num_items; item_id += step) {
                // read image pixel
                int size = batchsize;
                if (item_id + batchsize > num_items) {
                    size = num_items - item_id;
                }
                
                pixels_batch = pixels_all + item_id * rows * cols;
                labels_batch = labels_all + item_id * 10;
                
                nn->setTrainData(pixels_batch, labels_batch, size);
                nn->train(size);
            }
            
            nn->setTestData(pixels_test, labels_test, num_testItems);
            nn->realtest();
            printf("%s after 1 epoch, correct  is %.6f\n",timeStr().c_str(), nn->getCorrectRate());
            
            pass ++;
            if (nn->getCorrectRate() > 0.9) {
                break;
            }
            if (pass > 10000) {
                break;
            }
        }
        
        printf("train finish");
        delete[] pixels_all;
        delete[] labels_all;
        delete[] labels_all_original;
    }
    
//    void testMiniBatch(){
//        // Open files
//
//        std::ifstream test_image_file(test_image_filename, std::ios::in | std::ios::binary);
//        std::ifstream test_label_file(test_label_filename, std::ios::in | std::ios::binary);
//
//        if(!read_header(test_image_file, test_label_file, rows, cols, num_testItems, num_testLabels)){
//            return;
//        }
//
//        NaiveNeuralNet<char, double> * nn = new NaiveNeuralNet<char, double>();
//        nn->setDataStructure(rows * cols, 10);
//        if (this->neuralNet) {
//            delete this->neuralNet;
//        }
//        this->neuralNet = nn;
//
//        //test data
//        char* pixels_test = new char[rows * cols * num_testItems];
//
//        double* labels_test = new double[10 * num_testItems];
//        createTestData(test_image_file, test_label_file, pixels_test, labels_test, rows, cols, num_testItems);
//        nn->setTestData(pixels_test, labels_test, num_testItems);
//
//        char* pixels_batch = nullptr;
//        double* labels_batch= nullptr;  //0~9
//        char* labels_batch_original= nullptr ;
//
//        int pass = 0;
//        while (1) {
//            std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
//            std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
//
//            if(!read_header(image_file, label_file, rows, cols, num_items, num_labels)){
//                printf("read error");
//                return;
//            }
//
//            //train data
//            int batchsize = 2048;
//
//            char* pixels_batch = new char[rows * cols * batchsize];
//
//            double* labels_batch = new double[10 * batchsize]; //0~9
//
//            char* labels_batch_original = new char[ batchsize];
//
////            nn->lookAnInput(pixels_test, labels_test, 5000);
//
//            for (int item_id = 0; item_id < num_items; item_id += batchsize) {
//                // read image pixel
//                int size = batchsize;
//                if (item_id + batchsize > num_items) {
//                    size = num_items - item_id;
//                }
//                image_file.read(pixels_batch, rows * cols * size);
//                // read label
//                label_file.read(labels_batch_original, size);
//
//                for (int i=0; i<size; i++) {
//                    for (int shift = 0; shift < 10; shift ++) {
//                        labels_batch[i*10 + shift] =0; ;
//                    }
//
//                    labels_batch[i*10 + labels_batch_original[i]] = 1;
//                }
//
//                nn->setTrainData(pixels_batch, labels_batch, size);
//                nn->train(size);
//            }
//
//            nn->setTestData(pixels_test, labels_test, num_testItems);
//            nn->realtest();
//            printf("after 1 epoch, error is %.6f",nn->getCorrectRate());
//
//            pass ++;
//            if (nn->getCorrectRate() > 0.9) {
//                break;
//            }
//            if (pass > 10000) {
//                break;
//            }
//        }
//
//        printf("train finish");
//        delete[] pixels_batch;
//        delete[] labels_batch;
//        delete[] labels_batch_original;
//    }
    
    void testOneEPochLarge(){
        // Open files
        std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
        std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
        
        if(!read_header(image_file, label_file, rows, cols, num_items, num_labels)){
            return;
        }
        
        std::ifstream test_image_file(test_image_filename, std::ios::in | std::ios::binary);
        std::ifstream test_label_file(test_label_filename, std::ios::in | std::ios::binary);
        
        if(!read_header(test_image_file, test_label_file, rows, cols, num_testItems, num_testLabels)){
            return;
        }
        
        NaiveNeuralNet<char, double> * nn = new NaiveNeuralNet<char, double>();
        nn->setDataStructure(rows * cols, 10);
        if (this->neuralNet) {
            delete this->neuralNet;
        }
        this->neuralNet = nn;
        
        //test data
        char* pixels_test = new char[rows * cols * num_testItems];
        
        double* labels_test = new double[10 * num_testItems];
        createTestData(test_image_file, test_label_file, pixels_test, labels_test, rows, cols, num_testItems);
        nn->setTestData(pixels_test, labels_test, num_testItems);
        
        //train data
        int batchsize = 1000;
        
        char* pixels_batch = new char[rows * cols * batchsize];
        
        double* labels_batch = new double[10 * batchsize];
        
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
            for(int i=0;i<5000;i++){
                if(nn->train(batchsize)){
                    printf("train success");
                    break;
                }
                if ((i+1)%30 == 0) {
                    nn->realtest();
                }
                
                if (nn->getCorrectRate() > 0.9) {
                    printf("finish a set");
                    break;
                }
            }
            if (nn->getCorrectRate() > 0.98) {
                printf("train success");
                break;
            }
            
            nn->setTestData(pixels_test, labels_test, num_testItems);
            nn->realtest();
            printf("after 5 mini batches, error is %.6f",nn->getCorrectRate());
        }
        
        delete[] pixels_batch;
        delete[] labels_batch;
        delete[] labels_batch_original;
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
    
    void createTestData(std::ifstream & test_image_file, std::ifstream & test_label_file,char* pixels, double* labels, uint32_t rows,
                        uint32_t cols, uint32_t num_items){
        char labels_original;
        
        for (int item_id = 0; item_id < num_items; item_id += 1) {
            // read image pixel
            test_image_file.read(pixels+item_id*rows * cols, rows * cols);
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
