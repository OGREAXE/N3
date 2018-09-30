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
#include <vector>

enum ActivationFunction{
    //f(x) = max(0,x)
    //f'(x) = 0?0:1
    RecitifiedLinear,
    //f(x) = 1/(1+exp(-x))
    //f'(x) = f(x)(1-f(x))
    Sigmod,
    SoftMax,
};

enum CostFunction{
    // 1/2 * ∑[i]( (a[i] - t[i])^2 )
    // dE/dHout = (a[i] - t[i])
    MeanSquared,
    CrossEntrophy,
};

template<typename InputType, typename OutputType>
class NaiveNeuralNet{
private:
    unsigned int m_trainDataSetCount;
    InputType * m_trainDataSet;
    OutputType * m_labels;
    unsigned int m_testDataSetCount;
    InputType * m_testDataSet;
    OutputType * m_testLabels;
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
    float * m_previousHiddenWeights;
    float * m_previousOutputWeights;

    unsigned int m_numOutput;
    unsigned int m_numOutputWeights;
    float * m_outputWeights;
    float * m_outputs;
    
    float batch_size = 1000;
    
    ActivationFunction hiddenOuputActivation = RecitifiedLinear;
    ActivationFunction outputActivation = SoftMax;
    CostFunction costFunction = CrossEntrophy;
    
    //backpropagate
//    float * m_hiddenOuputErrorGradients;
//    float * m_outputErrorGradients;
//    float * m_hiddenErrorGradients;
    float * m_hiddenErrorGradientSum;
    float * m_outputWeightGradients;
    float * m_hiddenWeightGradients;
   
public:
    void setDataStructure(int inputNodeCount, int outputNodeCount){
        m_numInput = inputNodeCount + 1;
        m_numOutput = outputNodeCount;
        m_numHidden = (inputNodeCount + outputNodeCount)/2 + 1;
//        m_numHidden = inputNodeCount + 1;
        
        m_numHiddenWeights = m_numInput * (m_numHidden - 1);
        m_numOutputWeights = m_numHidden * m_numOutput;
        
        m_inputCache = new float[m_numInput];
        m_hiddenOuputs = new float[m_numHidden];
        m_outputs = new float[m_numOutput];
        
//        m_hiddenWeights = new float[m_numHiddenWeights];
//        m_previousHiddenWeights = new float[m_numHiddenWeights];
//        m_outputWeights = new float[m_numOutputWeights];
//        m_previousOutputWeights = new float[m_numOutputWeights];
        
        createDataArrayAndSetValue(m_hiddenWeights, m_numHiddenWeights,0);
        createDataArrayAndSetValue(m_previousHiddenWeights, m_numHiddenWeights,0);
        createDataArrayAndSetValue(m_outputWeights, m_numOutputWeights,0);
        createDataArrayAndSetValue(m_previousOutputWeights, m_numOutputWeights,0);
        
//        m_hiddenErrorGradientSum = new float[m_numHidden-1];
//        m_outputWeightGradients = new float[m_numOutputWeights];
//        m_hiddenWeightGradients = new float[m_numHiddenWeights];
        
        createDataArrayAndSetValue(m_hiddenErrorGradientSum, m_numHidden,0);
        createDataArrayAndSetValue(m_outputWeightGradients, m_numOutputWeights,0);
        createDataArrayAndSetValue(m_hiddenWeightGradients, m_numHiddenWeights,0);
        
        randomizeWeights();
        
    }
    
    void createDataArrayAndSetValue(float * & array, int count, float value){
        array = new float[count];
        for (int i=0; i< count; i++) {
            array[i] = value;
        }
    }
    
    void setTrainData(InputType * input, OutputType * label, int totalDataSetCount){
        m_trainDataSet = input;
        m_labels = label;
        m_trainDataSetCount = totalDataSetCount;
    }
    
    void setTestData(InputType * input, OutputType * label, int totalDataSetCount){
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
    
private:
    void randomizeWeights(){
        randomize(m_hiddenWeights, m_numHiddenWeights, m_numInput);
        randomize(m_outputWeights, m_numOutputWeights, m_numHidden);
    }
    
//    void randomize(float * data, int count, int layerInput){
//        float range = 1 / sqrt((float)layerInput);
//        uint32_t rangeInt = (uint32_t)(2000000.0f * range);
//
//        std::default_random_engine engine(time(nullptr));
//        std::uniform_int_distribution<> dis(0, rangeInt);
//        for(int i = 0;i<count;i++) {
//            float randomFloat = (float)(dis(engine)) - (float)(rangeInt / 2);
//            data[i] = randomFloat / 1000000.0f;
//        }
//    }
    
    void randomize(float * data, int count, int layerInput){
        for(int i = 0;i<count;i++) {
            float randomFloat = gaussrand();
            data[i] = randomFloat * 0.01;
            if (data[i] > 1) {
                int x= 0;
            }
        }
    }
    
    double gaussrand(){
        static double V1, V2, S;
        static int phase = 0;
        double X;
        if ( phase == 0 ) {
            do {
                double U1 = (double)rand() / RAND_MAX;
                double U2 = (double)rand() / RAND_MAX;
                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
                
            } while(S >= 1 || S == 0);
            
            X = V1 * sqrt(-2 * log(S) / S);
            
        } else
            X = V2 * sqrt(-2 * log(S) / S);
        phase = 1 - phase;
        return X;
    }
    
    bool infer(InputType * inputs){
        m_inputCache[0] = 1;
        
        for (int i=1; i<m_numInput; i++) {
//            m_inputCache[i] = ((float)inputs[i-1])/256.;
            
            m_inputCache[i] = (inputs[i]==0?0:1);
            
//            printf("%d",inputs[i]==0?0:6);
//            if ((i-1)%28==0) {
//                printf("\n");
//            }
        }
        
        int numHiddenOutputGreaterThan0 = 0;
        for (int i=1; i<m_numHidden; i++) {
            float netHiddenOutput = 0;
            for (int j=0; j<m_numInput; j++) {
                int index = getWeightIndex(j, i-1, m_numHidden-1);
                ////
                float d =  m_hiddenWeights[index] * m_inputCache[j];
                if (d>1) {
                    d++;
                    throw 0;
                }
                /////
                
                netHiddenOutput += m_hiddenWeights[index] * m_inputCache[j];
                
                /////?????
                float x =  m_hiddenWeights[index]* m_inputCache[j];
                if (x>1) {
                    float w = m_hiddenWeights[index];
                    float b = m_inputCache[index];
                    w++;
                    throw 0;
                }
                ///////????
            }
            
            m_hiddenOuputs[i] = activate(netHiddenOutput, hiddenOuputActivation);
            
            float d =  m_hiddenOuputs[i];
            
            if (d>1) {
                float w = netHiddenOutput;
                w++;
                throw 0;
            }
            if (d>0) {
                numHiddenOutputGreaterThan0 ++;
            }
        }
        m_hiddenOuputs[0] = 1;
        
        if (outputActivation == SoftMax) {
            m_hiddenOuputs[0] = 1;
            float max = -1;
            for (int i=1; i<m_numHidden; i++) {
                if (m_hiddenOuputs[i] > max) {
                    max = m_hiddenOuputs[i];
                }
            }
            max = 0;
            
            float sumExp = 0;
            std::vector<float> tmp(m_numOutput);
            for (int i=0; i<m_numOutput; i++) {
                float netOutput = 0;
                for (int j=0; j<m_numHidden; j++) {
                    int index = getWeightIndex(j, i, m_numOutput);
                    
                    //////////
                    float d =  m_outputWeights[index] * m_hiddenOuputs[j];
                    
                    if (d>1) {
                        float w = m_outputWeights[index];
                        float ho = m_hiddenOuputs[j];
                        d++;
                        throw 0;
                    }
                    //////////
                    
                    netOutput += m_outputWeights[index] * m_hiddenOuputs[j];
                }
                tmp[i] = netOutput- max;
                
                sumExp += exp(netOutput-max);
                if (sumExp == 0) {
                    i++;
                    throw 0;
                }
            }
            
            for (int i=0; i<m_numOutput; i++) {
                m_outputs[i] = exp(tmp[i])/sumExp;
            }
        }
        else{
            for (int i=0; i<m_numOutput; i++) {
                float netOutput = 0;
                for (int j=0; j<m_numHidden; j++) {
                    int index = getWeightIndex(j, i, m_numOutput);
                    netOutput += m_outputWeights[index] * m_hiddenOuputs[j];
                }
                
                m_outputs[i] = activate(netOutput, outputActivation);
            }
        }
        
        return true;
    }
    
    float activate(float input, ActivationFunction activation){
        switch (activation) {
            case Sigmod:{
                double v = 1./(1. + exp(-input));
                assert(v!=0);
                return v;
            }
            case RecitifiedLinear:
                return input > 0?input:0;
            default:
                break;
        }
        return 0;
    }
    
public:
    bool train(int batchSize){
        int epochCnt = 0;
//        float error = 0;
        this->batch_size = batchSize;
        while (epochCnt * batchSize < m_trainDataSetCount) {
            InputType * batchInput = m_trainDataSet + epochCnt * batchSize * (m_numInput-1);
            OutputType * batchLabel = m_labels + epochCnt * batchSize * m_numOutput;
            
            for (int i=0; i<batchSize; i++) {
                InputType * anInput = batchInput + i * (m_numInput-1);
                OutputType * anLabel = batchLabel + i * m_numOutput;
                
//                for (int k=0; k< 28 * 28; k++) {
//                    printf("%d",anInput[k]==0?0:6);
//                    if ((k-1)%28==0) {
//                        printf("\n");
//                    }
//                }
                
                infer(anInput);
                backpropagate(anLabel);
            }
            
            updateWeights();
            
            epochCnt ++;
            
            
//            printf("error %.8f\n",error);
        }
//        printf("train finish epoch");
        return false;
    }
    
    void test(){
        float error = 0;
        for (int i=0; i<m_testDataSetCount-1; i++) {
            InputType * anInput = m_testDataSet + i * (m_numInput-1);
            OutputType * anLabel = m_testLabels + i * m_numOutput;
            infer(anInput);
            error += cost(m_outputs, anLabel, costFunction);
        }
        error /= (float)m_testDataSetCount;
        
        if (error < 0.1) {
            printf("train finish, error %.3f",error);
        }
        printf("train finish epoch, error %.8f\n",error);
    }
    
    void realtest(){
        float right = 0;
        for (int i=0; i<m_testDataSetCount-1; i++) {
            InputType * anInput = m_testDataSet + i * (m_numInput-1);
            OutputType * anLabel = m_testLabels + i * m_numOutput;
            infer(anInput);
            
            int num = -1;
            for (int i=0; i<m_numOutput; i++) {
                if (anLabel[i] == 1) {
                    num = i;
                }
            }
            
            float max = -1;
            int infer = -1;
            for (int i=0; i<m_numOutput; i++) {
                float o0 = m_outputs[i];
                
                if (m_outputs[i] > max) {
                    max = m_outputs[i];
                    infer = i;
                }
            }
            if (infer >= 0) {
                if (infer == num) {
                    right ++;
                }
            }
            
//            for (int k=0; k< 28 * 28; k++) {
//                printf("%d",anInput[k]==0?0:6);
//                if ((k-1)%28==0) {
//                    printf("\n");
//                }
//            }
//
//            int x = 0;
        }
        float  correct = (float)right/m_testDataSetCount;
        
        printf("train finish epoch, right %.8f\n",correct);
    }
    
private:
    void updateWeights(){
        float learningRate = 0.05;
        float a = 0.2;
        int c = 0;
        for (int i=0; i<m_numInput; i++) {
            for (int j=0; j<m_numHidden-1; j++) {
                int index = getWeightIndex(i, j, m_numHidden-1);
                
                float weight = m_hiddenWeights[index];
                float grad = m_hiddenWeightGradients[index]/batch_size;
                if (grad !=0) {
                    c++;
                }
                
//                m_hiddenWeights[index] -= learningRate * m_hiddenWeightGradients[index]/batch_size;
                
                m_hiddenWeights[index] = m_hiddenWeights[index] - (1-a)*learningRate * grad * m_inputCache[i] + a *(m_hiddenWeights[index] -  m_previousHiddenWeights[index]);
                
                m_previousHiddenWeights[index] = weight;
                if (m_hiddenWeightGradients[index]/batch_size > 1) {
                    float k = m_hiddenWeightGradients[index]/batch_size;
                    float x= m_hiddenWeights[index];
                    x++;
                    throw 0;
                }
                m_hiddenWeightGradients[index] = 0;
            }
        }
        
        for (int i=0; i<m_numHidden; i++) {
            for (int j=0; j<m_numOutput; j++) {
                int index = getWeightIndex(i, j, m_numOutput);
                
                float weight = m_outputWeights[index];
                float grad = m_outputWeightGradients[index]/batch_size;
                if (grad !=0) {
                    c++;
                }
                
//                m_outputWeights[index] -= learningRate *m_outputWeightGradients[index];
                m_outputWeights[index] = m_outputWeights[index] - (1-a)*learningRate * grad * m_hiddenOuputs[i] + a *(m_outputWeights[index] -  m_previousOutputWeights[index]);
                
                m_previousOutputWeights[index] = weight;
                
                if (m_outputWeightGradients[index]/batch_size > 1) {
                    float k = m_outputWeightGradients[index]/batch_size;
                    float x= m_outputWeights[index];
                    x++;
                    throw 0;
                }
                
                m_outputWeightGradients[index] = 0;
                
                
            }
        }
        
        for (int j=0; j<m_numHidden; j++) {
            m_hiddenErrorGradientSum[j]  = 0;
        }
        
        c++;
//        printf("none zero weight gradient count is %d",c);
    }
    
    //backpropagate one set in a batch
    bool backpropagate(OutputType * labels){
        //update output layer weights
        for (int i=0; i<m_numOutput; i++) {
            float dOut_dnet = cost_derivate(m_outputs[i] , labels[i], costFunction) * derivate(m_outputs[i], outputActivation);
            
            for (int j=0; j<m_numHidden; j++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_outputWeightGradients[outWeightIndex]  += dOut_dnet * m_hiddenOuputs[j];
            }
        }
        
        //calculate hidden layer gradient
        for (int j=1; j<m_numHidden; j++) {
            for (int i=0; i<m_numOutput; i++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_hiddenErrorGradientSum[j-1]  += cost_derivate(m_outputs[i] , labels[i], costFunction) * derivate(m_outputs[i], outputActivation) * m_outputWeights[outWeightIndex];
            }
        }
        
        //update hidden layer weights
        for (int i=0; i<m_numHidden-1; i++) {
            for (int j=0; j<m_numInput; j++) {
                int hiddenWeightIndex = getWeightIndex(j, i, m_numHidden-1);
                m_hiddenWeightGradients[hiddenWeightIndex] += m_hiddenErrorGradientSum[i] *derivate(m_hiddenOuputs[i], hiddenOuputActivation)* m_hiddenWeights[hiddenWeightIndex];
                
                if (m_hiddenWeightGradients[hiddenWeightIndex]/batch_size > 1) {
                    float heg = m_hiddenErrorGradientSum[i];
                    float d = derivate(m_hiddenOuputs[i], hiddenOuputActivation);
                    float weight  = m_hiddenWeights[hiddenWeightIndex];
                    int sss = 0;
                }
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
                return output * (1.-output);
                break;
            case RecitifiedLinear:
                return output == 0 ? 0 : 1;
            case SoftMax:
                return 1;
            default:
                break;
        }
        return 0;
    }
    
    float cost(float * output, OutputType * target, CostFunction costFunction){
        switch (costFunction) {
            case MeanSquared:
            {
                float err = 0;
                for (int i=0; i<m_numOutput; i++) {
                    err += (output[i] - (float)target[i]) * (output[i] - (float)target[i]) /2.;
                }
                return err;
            }
            case CrossEntrophy:
            {
                if (outputActivation == SoftMax) {
                    float sum = 0;
                    for (int i=0; i<m_numOutput; i++) {
                        float p = output[i];
                        float t = target[i];
                        sum += t*log(p);
                    }
                    return -sum;
                }
                else{
                    float sum = 0;
                    for (int i=0; i<m_numOutput; i++) {
                        float r = output[i];
                        float t = target[i];
                        sum += t*log(r) + (1-t)*log(1-r);
                    }
                    return -sum;
                }
            }
            default:
                break;
        }
        return 0;
    }
    
    float cost_derivate(float output, OutputType target, CostFunction costFunction){
        float t = target;
        switch (costFunction) {
            case MeanSquared:
                return (output - (float)target);
            case CrossEntrophy:
            {
                if (outputActivation == SoftMax) {
                    return output-t;
                }
                else{
                    return (output - t) / ((1. - output) * output);
                }
            }
            default:
                break;
        }
        return 0;
    }
};

#endif /* NextNeuralNet_hpp */
