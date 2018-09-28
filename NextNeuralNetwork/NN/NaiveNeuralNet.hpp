//
//  NaiveNeuralNet.hpp
//  NextNeuralNetwork
//
//  Created by 梁志远 on 2018/6/18.
//  Copyright © 2018 Ogreaxe. All rights reserved.
//

#ifndef NextNeuralNet_hpp
#define NextNeuralNet_hpp

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>

enum ActivationFunction{
    //f(x) = max(0,x)
    //f'(x) = 0?0:1
    RecitifiedLinear,
    //f(x) = 1/(1+exp(-x))
    //f'(x) = f(x)(1-f(x))
    Sigmod,
};

enum CostFunction{
    // 1/2 * ∑[i]( (a[i] - t[i])^2 )
    // dE/dHout = (a[i] - t[i])
    MeanSquared,
};

class NaiveNeuralNet{
private:
    unsigned int m_trainDataSetCount;
    float * m_trainDataSet;
    float * m_labels;
    unsigned int m_testDataSetCount;
    float * m_testDataSet;
    float * m_testLabels;
    //number of inputs, INCLUDE bias
    unsigned int m_numInput;
    unsigned int m_numInputWeights;
    //input[0] is 1 to multiple bias
    float * m_inputCache;
    //inputWeights[0] for bias
    float * m_inputWeights;
    
    //number of hidden nodes, INCLUDE bias
    unsigned int m_numHidden;
    unsigned int m_numHiddenWeights;
    float * m_hiddenWeights;
    float * m_hiddenOuputs;

    unsigned int m_numOutput;
    unsigned int m_numOutputWeights;
    float * m_outputWeights;
    float * m_outputs;
    
    ActivationFunction hiddenOuputActivation;
    ActivationFunction outputActivation;
    CostFunction costFunction;
    
    //backpropagate
//    float * m_hiddenOuputErrorGradients;
//    float * m_outputErrorGradients;
//    float * m_hiddenErrorGradients;
    float * m_hiddenErrorGradientSum;
    float * m_outputWeightGradients;
    float * m_hiddenWeightGradients;
    
    void setDataStructure(int inputNodeCount, int outputNodeCount){
        m_numInput = inputNodeCount + 1;
        m_numOutput = outputNodeCount;
        m_numHidden = (inputNodeCount + outputNodeCount)/2 + 1;
        
        m_numHiddenWeights = m_numInput * (m_numHidden - 1);
        m_numOutputWeights = m_numHidden * m_numOutput;
        
        m_inputCache = new float[m_numInput];
        m_hiddenOuputs = new float[m_numHidden];
        m_outputs = new float[m_numOutput];
        
        m_hiddenWeights = new float[m_numHiddenWeights];
        m_outputWeights = new float[m_numOutputWeights];
    }
    
    void setTrainData(float * input, float * label, int totalDataSetCount){
        m_trainDataSet = input;
        m_labels = label;
        m_trainDataSetCount = totalDataSetCount;
    }
    
    void setTestData(float * input, float * label, int totalDataSetCount){
        m_testDataSet = input;
        m_testLabels = label;
        m_testDataSetCount = totalDataSetCount;
    }
    
//    void prepareData(int dataSetOffset){
//        if (m_inputCache) {
//            delete [] m_inputCache;
//        }
//
//        m_inputCache = new float[m_numInput];
//        m_inputCache[0] = 1;
//        for (int i=0; i<m_numInput; i++) {
//            m_inputCache[i+1] = m_inputCache[dataSetOffset + i];
//        }
//
//        m_numHidden = 2 * m_numInput + 1;
//    }
    
    void randomizeWeights(){
        randomize(m_hiddenWeights, m_numHiddenWeights, m_numInput);
        randomize(m_outputWeights, m_numOutputWeights, m_numHidden);
    }
    
    void randomize(float * data, int count, int layerInput){
        float range = 1 / sqrt((float)layerInput);
        uint32_t rangeInt = (uint32_t)(2000000.0f * range);
        
        std::default_random_engine engine(time(nullptr));
        std::uniform_int_distribution<> dis(0, rangeInt);
        for(int i = 0;i<count;i++) {
            float randomFloat = (float)(dis(engine)) - (float)(rangeInt / 2);
            data[i] = randomFloat / 1000000.0f;
        }
    }
    
    bool infer(float * inputs){
        m_inputCache[0] = 1;
        
        for (int i=1; i<m_numInput; i++) {
            m_inputCache[i] = inputs[i-1];
        }
        
        for (int i=1; i<m_numHidden; i++) {
            float netHiddenOutput = 0;
            for (int j=0; j<m_numInput; j++) {
                netHiddenOutput += m_hiddenWeights[getWeightIndex(j, i-1, m_numHidden-1)] * m_inputCache[j];
            }
            
            m_hiddenOuputs[i] = activate(netHiddenOutput, hiddenOuputActivation);
        }
        m_hiddenOuputs[0] = 1;
        
        for (int i=0; i<m_numOutput; i++) {
            float netOutput = 0;
            for (int j=0; j<m_numHidden; j++) {
                netOutput += m_outputWeights[getWeightIndex(j, i, m_numOutput)] * m_hiddenOuputs[j];
            }
            
            m_outputs[i] = activate(netOutput, outputActivation);
        }
        
        return true;
    }
    
    float activate(float input, ActivationFunction activation){
        switch (activation) {
            case Sigmod:
                return 1./(1. + exp(-input));
            default:
                break;
        }
        return 0;
    }
    
    bool train(int batchSize){
        int epochCnt = 0;
        float error = 0;
        while (epochCnt * batchSize < m_trainDataSetCount) {
            float * batchInput = m_trainDataSet + epochCnt * batchSize * m_numInput;
            float * batchLabel = m_labels + epochCnt * batchSize * m_numOutput;
            for (int i=0; i<batchSize; i++) {
                float * anInput = batchInput + i * m_numInput;
                float * anLabel = batchLabel + i * m_numOutput;
                
                infer(anInput);
                backpropagate(anInput, anLabel);
            }
            
            updateWeights();
            
            epochCnt ++;
            
            for (int i=0; i<m_testDataSetCount; i++) {
                float * anInput = m_testDataSet + i * m_numInput;
                float * anLabel = m_testLabels + i * m_numOutput;
                infer(anInput);
                error += cost(m_outputs, anLabel, costFunction);
            }
            error /= (float)m_testDataSetCount;
            
            if (error < 0.1) {
                break;
            }
        }
        printf("train finish, error %.3f",error);
        return true;
    }
    
    void updateWeights(){
        for (int i=0; i<m_numInput; i++) {
            for (int j=0; j<m_numHidden; j++) {
                int index = getWeightIndex(i, j, m_numHidden);
                m_hiddenWeights[index] += m_hiddenWeightGradients[index];
                m_hiddenWeightGradients[index] = 0;
            }
        }
        
        for (int i=0; i<m_numHidden; i++) {
            for (int j=0; j<m_numOutput; j++) {
                int index = getWeightIndex(i, j, m_numOutput);
                m_outputWeights[index] += m_outputWeightGradients[index];
                m_outputWeightGradients[index] = 0;
            }
        }
    }
    
    //backpropagate one set in a batch
    bool backpropagate(float * inputs, float * labels){
        //update output layer weights
        for (int i=0; i<m_numOutput; i++) {
            float dOut_dnet = cost_derivate(m_outputs[i] , labels[i], MeanSquared) * derivate(m_outputs[i], outputActivation);
            
            for (int j=0; j<m_numHidden; j++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_outputWeightGradients[outWeightIndex]  += dOut_dnet * m_hiddenOuputs[i];
            }
        }
        
        //calculate hidden layer gradient
        for (int j=1; j<m_numHidden; j++) {
            for (int i=0; i<m_numOutput; i++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_hiddenErrorGradientSum[j]  += cost_derivate(m_outputs[i] , labels[i], MeanSquared) * derivate(m_outputs[i], outputActivation) * m_outputWeights[outWeightIndex];
            }
        }
        
        //update hidden layer weights
        for (int i=0; i<m_numHidden-1; i++) {
            for (int j=0; j<m_numInput; j++) {
                int hiddenWeightIndex = getWeightIndex(j, i, m_numHidden-1);
                m_hiddenWeightGradients[hiddenWeightIndex] += m_hiddenErrorGradientSum[i] *derivate(m_hiddenOuputs[i], hiddenOuputActivation)* m_hiddenWeights[hiddenWeightIndex];
            }
            
        }
        
        return true;
    }
    
    int getWeightIndex(int inNodeIndex,int outNodeIndex, int outNodeCount){
        return inNodeIndex * outNodeCount + outNodeIndex;
    }
    
    
    float derivate(float output, ActivationFunction activation){
        switch (activation) {
            case Sigmod:
                return output * (1-output);
                break;
                
            default:
                break;
        }
        return 0;
    }
    
    float cost(float * output, float * target, CostFunction costFunction){
        switch (costFunction) {
            case MeanSquared:
            {
                float err = 0;
                for (int i=0; i<m_numOutput; i++) {
                    err += (output - target) * (output - target) /2.;
                }
                return err;
            }
                
            default:
                break;
        }
        return 0;
    }
    
    float cost_derivate(float output, float target, CostFunction costFunction){
        switch (costFunction) {
            case MeanSquared:
                return (output - target);
            default:
                break;
        }
        return 0;
    }
};

#endif /* NextNeuralNet_hpp */
