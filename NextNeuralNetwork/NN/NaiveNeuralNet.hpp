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
#include <limits>

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
    double * m_inputCache;
    //inputWeights[0] for bias
    double * m_inputWeights;
    
    //number of hidden nodes, INCLUDE bias
    unsigned int m_numHidden;
    unsigned int m_numHiddenWeights;
    double * m_hiddenWeights;
    double * m_hiddenOuputs;
    double * m_previousHiddenWeights;
    double * m_previousOutputWeights;

    unsigned int m_numOutput;
    unsigned int m_numOutputWeights;
    double * m_outputWeights;
    double * m_outputs;
    
    double * m_outputDerivative;
    
    double batch_size = 1000;
    double m_learningRate = 0.0007;
    double m_momentum = 0.95;
    
    double m_epochCount = 0;
    double m_correctRate = 0;
    
    double m_maxDelta = 0;
    
    double m_mxHiddenOutput = 0;
    
    ActivationFunction hiddenOuputActivation = RecitifiedLinear;
    ActivationFunction outputActivation = SoftMax;
    CostFunction costFunction = CrossEntrophy;
    
    //backpropagate
//    double * m_hiddenOuputErrorGradients;
//    double * m_outputErrorGradients;
//    double * m_hiddenErrorGradients;
    double * m_hiddenErrorGradientSum;
    double * m_outputWeightGradients;
    double * m_hiddenWeightGradients;
    
    
    int ** m_inputWeightIndexCache;
    int ** m_outputWeightIndexCache;
   
public:
    ~NaiveNeuralNet(){
        release2dWeightIndexCache(m_inputWeightIndexCache,m_numInput);
        release2dWeightIndexCache(m_outputWeightIndexCache,m_numHidden);
    }
    
    double getCorrectRate(){
        return m_correctRate;
    }
    
    void setDataStructure(int inputNodeCount, int outputNodeCount){
        release2dWeightIndexCache(m_inputWeightIndexCache,m_numInput);
        release2dWeightIndexCache(m_outputWeightIndexCache,m_numHidden);
        
        m_epochCount = 0;
        m_correctRate = 0;
        
        m_numInput = inputNodeCount + 1;
        m_numOutput = outputNodeCount;
        m_numHidden = (inputNodeCount + outputNodeCount)/2 + 1;
//        m_numHidden = inputNodeCount + 1;
        
        m_numHiddenWeights = m_numInput * (m_numHidden - 1);
        m_numOutputWeights = m_numHidden * m_numOutput;
        
        m_inputCache = new double[m_numInput];
        m_hiddenOuputs = new double[m_numHidden];
        m_outputs = new double[m_numOutput];
        
        m_outputDerivative = new double[m_numOutput];
        
//        m_hiddenWeights = new double[m_numHiddenWeights];
//        m_previousHiddenWeights = new double[m_numHiddenWeights];
//        m_outputWeights = new double[m_numOutputWeights];
//        m_previousOutputWeights = new double[m_numOutputWeights];
        
        createDataArrayAndSetValue(m_hiddenWeights, m_numHiddenWeights,0);
        createDataArrayAndSetValue(m_previousHiddenWeights, m_numHiddenWeights,0);
        createDataArrayAndSetValue(m_outputWeights, m_numOutputWeights,0);
        createDataArrayAndSetValue(m_previousOutputWeights, m_numOutputWeights,0);
        
//        m_hiddenErrorGradientSum = new double[m_numHidden-1];
//        m_outputWeightGradients = new double[m_numOutputWeights];
//        m_hiddenWeightGradients = new double[m_numHiddenWeights];
        
        createDataArrayAndSetValue(m_hiddenErrorGradientSum, m_numHidden,0);
        createDataArrayAndSetValue(m_outputWeightGradients, m_numOutputWeights,0);
        createDataArrayAndSetValue(m_hiddenWeightGradients, m_numHiddenWeights,0);
        
        randomizeWeights();
        
        //weight index cache
        m_inputWeightIndexCache = create2dWeightIndexCache(m_numInput,m_numHidden-1);
        m_outputWeightIndexCache = create2dWeightIndexCache(m_numHidden,m_numOutput);
    }
    
    int ** create2dWeightIndexCache(int inputCnt, int outCnt){
        int ** row = new int* [inputCnt];
        for (int i = 0; i<inputCnt; i++) {
            int * col = new int[outCnt];
            row[i] = col;
            for (int j = 0; j<outCnt; j++) {
                col[j] = i * outCnt + j;
            }
        }
        return row;
    }
    
    void release2dWeightIndexCache(int ** cache, int inputCnt){
        if (!cache) {
            return;
        }
        
        for (int i = 0; i<inputCnt; i++) {
            delete [] cache[i];
        }
        delete [] cache;
    }
    
    void createDataArrayAndSetValue(double * & array, int count, double value){
        array = new double[count];
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
//        m_inputCache = new double[m_numInput];
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
    
//    void randomize(double * data, int count, int layerInput){
//        double range = 1 / sqrt((double)layerInput);
//        uint32_t rangeInt = (uint32_t)(2000000.0f * range);
//
//        std::default_random_engine engine(time(nullptr));
//        std::uniform_int_distribution<> dis(0, rangeInt);
//        for(int i = 0;i<count;i++) {
//            double randomdouble = (double)(dis(engine)) - (double)(rangeInt / 2);
//            data[i] = randomdouble / 1000000.0f;
//        }
//    }
    
    void randomize(double * data, int count, int layerInput){
        for(int i = 0;i<count;i++) {
            double randomdouble = gaussrand();
            data[i] = randomdouble * 0.01;
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
            m_inputCache[i] = ((double)((unsigned char)inputs[i-1]))/256.;
            
//            m_inputCache[i] = (inputs[i]==0?0:1);
            
//            printf("%d",inputs[i]==0?0:6);
//            if ((i-1)%28==0) {
//                printf("\n");
//            }
        }
        
        int numHiddenOutputGreaterThan0 = 0;
        for (int i=1; i<m_numHidden; i++) {
            double netHiddenOutput = 0;
            for (int j=0; j<m_numInput; j++) {
//                int index = getWeightIndex(j, i-1, m_numHidden-1);
                int index = getInputWeightIndex(j, i-1);
                ////
//                double d =  m_hiddenWeights[index] * m_inputCache[j];
//                if (d>1) {
//                    d++;
//                    throw 0;
//                }
                /////
                
                netHiddenOutput += m_hiddenWeights[index] * m_inputCache[j];
                
                /////?????
//                double x =  m_hiddenWeights[index]* m_inputCache[j];
//                if (x>1) {
//                    double w = m_hiddenWeights[index];
//                    double b = m_inputCache[index];
//                    w++;
//                    throw 0;
//                }
                ///////????
            }
            
            m_hiddenOuputs[i] = activate(netHiddenOutput, hiddenOuputActivation);
            
            ////
            //for ReLU, output greater than 1 is possible
//            double d =  m_hiddenOuputs[i];
//            if (d>1) {
//                double w = netHiddenOutput;
//                w++;
//                throw 0;
//            }
//            ////
//            if (d>0) {
//                numHiddenOutputGreaterThan0 ++;
//            }
        }
        m_hiddenOuputs[0] = 1;
        
        m_mxHiddenOutput = 0;
        for (int i=0; i<m_numHidden; i++) {
            if (m_hiddenOuputs[i] > m_mxHiddenOutput) {
                m_mxHiddenOutput = m_hiddenOuputs[i];
            }
        }
        
        if (outputActivation == SoftMax) {
            double max = std::numeric_limits<double>::lowest();
            
            double sumExp = 0;
            std::vector<double> tmp(m_numOutput);
            
//            if (m_epochCount >= 150 && getCorrectRate() < 0.2) {
//                int i=0;
//            }
            for (int i=0; i<m_numOutput; i++) {
                double netOutput = 0;
                for (int j=0; j<m_numHidden; j++) {
//                    int index = getWeightIndex(j, i, m_numOutput);
                    int index = getOutputWeightIndex(j, i);
                    
                    //////////
//                    double d =  m_outputWeights[index] * m_hiddenOuputs[j];
//
//                    if (d>1) {
//                        double w = m_outputWeights[index];
//                        double ho = m_hiddenOuputs[j];
//                        d++;
//                        throw 0;
//                    }
                    //////////
                    
                    netOutput += m_outputWeights[index] * m_hiddenOuputs[j];
                }
                tmp[i] = netOutput;
 //               checkoverflow(netOutput);
                if (netOutput > max) {
                    max = netOutput;
                }
            }
            
            for (int i=0; i<m_numOutput; i++) {
                float oldsum = sumExp;
                sumExp += exp(tmp[i]-max);
 //               checkoverflow(sumExp);
            }
            
            if (sumExp == 0) {
                throw 0;
            }
            
            for (int i=0; i<m_numOutput; i++) {
                m_outputs[i] = exp(tmp[i]-max)/sumExp;
//                checkoverflow(m_outputs[i]);
            }
        }
        else{
            for (int i=0; i<m_numOutput; i++) {
                double netOutput = 0;
                for (int j=0; j<m_numHidden; j++) {
//                    int index = getWeightIndex(j, i, m_numOutput);
                    int index = getOutputWeightIndex(j, i);
                    netOutput += m_outputWeights[index] * m_hiddenOuputs[j];
                }
                
                m_outputs[i] = activate(netOutput, outputActivation);
            }
        }
        
        return true;
    }
    
    double activate(double input, ActivationFunction activation){
        switch (activation) {
            case Sigmod:{
                double v = 1./(1. + exp(-input));
                //assert(v!=0);
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
//        double error = 0;
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
        m_epochCount += epochCnt;
        return false;
    }
    
    void test(){
        double error = 0;
        for (int i=0; i<m_testDataSetCount-1; i++) {
            InputType * anInput = m_testDataSet + i * (m_numInput-1);
            OutputType * anLabel = m_testLabels + i * m_numOutput;
            infer(anInput);
            error += cost(m_outputs, anLabel, costFunction);
        }
        error /= (double)m_testDataSetCount;
        
        if (error < 0.1) {
            printf("train finish, error %.3f",error);
        }
        printf("train finish epoch, error %.8f\n",error);
    }
    
    void realtest(){
        double right = 0;
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
            
            double max = -1;
            int infer = -1;
            for (int i=0; i<m_numOutput; i++) {
                double o0 = m_outputs[i];
                
                if (m_outputs[i] > max) {
                    max = m_outputs[i];
                    infer = i;
                }
            }
            
            if (infer >= 0) {
                if (infer == num) {
                    right ++;
                }
                else {
//                    lookAnInput(anInput,anLabel,0);
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
        double  correct = (double)right/m_testDataSetCount;
//        if (correct < m_correctRate) {
//            m_learningRate = m_learningRate * 0.2;
//        }
        m_correctRate = correct;
//        printf("train finish epoch %f, right %.8f\n",m_epochCount,correct);
    }
    
private:
    void updateWeights(){
        
//        if ((long long)(m_epochCount+1) % 40==0 && m_learningRate > 0.05) {
//            m_learningRate = m_learningRate * 0.4;
//        }
    
        double momentum = m_momentum;
        int c = 0;
        for (int i=0; i<m_numInput; i++) {
            for (int j=0; j<m_numHidden-1; j++) {
//                int index = getWeightIndex(i, j, m_numHidden-1);
                int index = getInputWeightIndex(i, j);
                
                double weight = m_hiddenWeights[index];
                double grad = m_hiddenWeightGradients[index]/batch_size;
                if (grad !=0) {
                    c++;
                }
                
//                m_hiddenWeights[index] -= learningRate * m_hiddenWeightGradients[index]/batch_size;
                double delta = - (1-momentum)*m_learningRate * grad * m_inputCache[i] + momentum *(m_hiddenWeights[index] -  m_previousHiddenWeights[index]);
//                checkoverflow(m_hiddenWeights[index]);
//                delta = clipDelta(delta);
                m_hiddenWeights[index] = m_hiddenWeights[index] + delta ;
                
                if(m_maxDelta < abs(delta)){
                    m_maxDelta = abs(delta);
                }
                
                m_previousHiddenWeights[index] = weight;
//                if (m_hiddenWeightGradients[index]/batch_size > 1) {
//                    double k = m_hiddenWeightGradients[index]/batch_size;
//                    double x= m_hiddenWeights[index];
//                    x++;
//                    throw 0;
//                }
                m_hiddenWeightGradients[index] = 0;
            }
        }
        
        for (int i=0; i<m_numHidden; i++) {
            for (int j=0; j<m_numOutput; j++) {
//                int index = getWeightIndex(i, j, m_numOutput);
                int index = getOutputWeightIndex(i, j);
                
                double weight = m_outputWeights[index];
                double grad = m_outputWeightGradients[index]/batch_size;
                if (grad !=0) {
                    c++;
                }
                
//                m_outputWeights[index] -= learningRate *m_outputWeightGradients[index];
                double delta = - (1-momentum)*m_learningRate * grad * m_hiddenOuputs[i] + momentum *(m_outputWeights[index] -  m_previousOutputWeights[index]);
//                m_outputWeights[index] = m_outputWeights[index] - (1-momentum)*learningRate * grad * m_hiddenOuputs[i] + momentum *(m_outputWeights[index] -  m_previousOutputWeights[index]);
//                checkoverflow(m_outputWeights[index]);
                m_outputWeights[index] = m_outputWeights[index] + delta;
                
                m_previousOutputWeights[index] = weight;
                
//                if (m_outputWeightGradients[index]/batch_size > 1) {
//                    double k = m_outputWeightGradients[index]/batch_size;
//                    double x= m_outputWeights[index];
//                    x++;
//                    throw 0;
//                }
                
                m_outputWeightGradients[index] = 0;
                
                
            }
        }
        
        for (int j=0; j<m_numHidden; j++) {
            m_hiddenErrorGradientSum[j]  = 0;
        }
        
        c++;
//        printf("none zero weight gradient count is %d",c);
    }
    
    inline bool checkoverflow(double value){
        if (std::isnan(value)) {
            throw 0;
        }
        return true;
//        return !std::isnan(value);
    }
    
    double clipDelta(double val){
        
        if (val < -1) {
            return -0.1;
        }
        else if (val > 1) {
            return 0.1;
        }
        return val;
    }

    //backpropagate one set in a batch
    bool backpropagate(OutputType * labels){
        //update output layer weights
        for (int i=0; i<m_numOutput; i++) {
//            double dOut_dnet = cost_derivate(m_outputs[i] , labels[i], costFunction) * derivate(m_outputs[i], outputActivation);
            
            m_outputDerivative[i] = cost_derivate(m_outputs[i] , labels[i], costFunction) * derivate(m_outputs[i], outputActivation);
            
            for (int j=0; j<m_numHidden; j++) {
//                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                int outWeightIndex = getOutputWeightIndex(j, i);
                m_outputWeightGradients[outWeightIndex]  += m_outputDerivative[i] * m_hiddenOuputs[j];
 //               checkoverflow(m_outputWeightGradients[outWeightIndex]);
            }
        }
        
        //calculate hidden layer gradient
        for (int j=1; j<m_numHidden; j++) {
            for (int i=0; i<m_numOutput; i++) {
//                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                int outWeightIndex = getOutputWeightIndex(j, i);
                m_hiddenErrorGradientSum[j-1]  += m_outputDerivative[i] * m_outputWeights[outWeightIndex];
            }
        }
        
        //update hidden layer weights
        for (int i=0; i<m_numHidden-1; i++) {
			double tmpHiddenDeriv = m_hiddenErrorGradientSum[i] * derivate(m_hiddenOuputs[i], hiddenOuputActivation);
            for (int j=0; j<m_numInput; j++) {
//                int hiddenWeightIndex = getWeightIndex(j, i, m_numHidden-1);
                int hiddenWeightIndex = getInputWeightIndex(j, i);
                //m_hiddenWeightGradients[hiddenWeightIndex] += m_hiddenErrorGradientSum[i] *derivate(m_hiddenOuputs[i], hiddenOuputActivation)* m_hiddenWeights[hiddenWeightIndex];
				m_hiddenWeightGradients[hiddenWeightIndex] += tmpHiddenDeriv* m_hiddenWeights[hiddenWeightIndex];
            }
            
        }
        
        return true;
    }
    
    inline int getWeightIndex(int inNodeIndex,int outNodeIndex, int outNodeCount){
        return inNodeIndex * outNodeCount + outNodeIndex;
    }
    
    inline int getInputWeightIndex(int inNodeIndex,int outNodeIndex){
        return m_inputWeightIndexCache[inNodeIndex][outNodeIndex];
    }
    
    inline int getOutputWeightIndex(int inNodeIndex,int outNodeIndex){
        return m_outputWeightIndexCache[inNodeIndex][outNodeIndex];
    }
    
    
    double derivate(double output, ActivationFunction activation){
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
    
    double cost(double * output, OutputType * target, CostFunction costFunction){
        switch (costFunction) {
            case MeanSquared:
            {
                double err = 0;
                for (int i=0; i<m_numOutput; i++) {
                    err += (output[i] - (double)target[i]) * (output[i] - (double)target[i]) /2.;
                }
                return err;
            }
            case CrossEntrophy:
            {
                if (outputActivation == SoftMax) {
                    double sum = 0;
                    for (int i=0; i<m_numOutput; i++) {
                        double p = output[i];
                        double t = target[i];
                        sum += t*log(p);
                    }
                    return -sum;
                }
                else{
                    double sum = 0;
                    for (int i=0; i<m_numOutput; i++) {
                        double r = output[i];
                        double t = target[i];
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
    
    double cost_derivate(double output, OutputType target, CostFunction costFunction){
        double t = target;
        switch (costFunction) {
            case MeanSquared:
                return (output - (double)target);
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
 
public:
    void lookAnInput(InputType * inputo, OutputType * labelo, int offset){
        InputType * input = inputo + 28 * 28 * offset;
        OutputType * anLabel = labelo + 10 * offset;
        int num = -1;
        for (int i=0; i<m_numOutput; i++) {
            if (anLabel[i] == 1) {
                num = i;
            }
        }
        
        printf("number is %d",num);
        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                if (input[i+28 * j] != 0) {
                    printf("*");
                }
                else {
                    printf("_");
                }
            }
            printf("\n");
        }
    }
};

#endif /* NextNeuralNet_hpp */
