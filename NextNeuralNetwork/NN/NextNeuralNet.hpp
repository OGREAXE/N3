//
//  NextNeuralNet.hpp
//  NextNeuralNetwork
//
//  Created by 梁志远 on 2018/6/18.
//  Copyright © 2018 Ogreaxe. All rights reserved.
//

#ifndef NextNeuralNet_hpp
#define NextNeuralNet_hpp

#include <stdio.h>

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

class NextNeuralNet{
private:
    //number of inputs, INCLUDE bias
    unsigned int m_numInput;
    //input[0] is 1 to multiple bias
    float * m_inputCache;
    //inputWeights[0] for bias
    float * m_inputWeights;
    
    //number of hidden nodes, INCLUDE bias
    unsigned int m_numHidden;
    float * m_hiddenWeights;
    float * m_hiddenOuputs;
    
    ActivationFunction hiddenActivation;
    
    unsigned int m_numOutput;
    float * m_outputWeights;
//    float * m_outputWeightGradients;
    float * m_outputs;
    
    ActivationFunction outputActivation;
    
    CostFunction costFunction;
    
    //backpropagate
    float * m_hiddenOuputErrorGradients;
    float * m_outputErrorGradients;
    float * m_hiddenErrorGradients;
    float * m_hiddenErrorGradientSum;
    
    float * m_outputWeightGradients;
    float * m_hiddenWeightGradients;
    
    bool infer(float * inputs){
        m_inputCache[0] = 1;
        
        for (int i=1; i<m_numInput; i++) {
            m_inputCache[i] = inputs[i-1];
        }
        
        for (int i = 0; i<m_numHidden; i++) {
            m_hiddenOuputs[i] = 0;
        }
        
        m_hiddenOuputs[0] = 1;
        
        for (int i=1; i<m_numHidden; i++) {
            float netHiddenOutput = 0;
            for (int j=0; j<m_numInput; j++) {
                netHiddenOutput += m_hiddenWeights[i+ j * m_numInput] * m_inputCache[j];
            }
            
            m_hiddenOuputs[i] = activate(netHiddenOutput, hiddenActivation);
        }
        
        for (int i=1; i<m_numOutput; i++) {
            float netOutput = 0;
            for (int j=0; j<m_numHidden; j++) {
                netOutput += m_outputWeights[i+ j * m_numHidden] * m_hiddenOuputs[j];
            }
            
            m_outputs[i] = activate(netOutput, outputActivation);
        }
        
        return true;
    }
    
    float activate(float input, ActivationFunction activation){
        return 0;
    }
    
    bool backpropagate(float ** inputs, float ** labels,int trainSetSize, int batchSize){
        int time = 0;
        while (time * batchSize < trainSetSize) {
            for (int i=0; i<batchSize; i++) {
                float * anInput = inputs[time * batchSize + i];
                float * anLabel = labels[time * batchSize + i];
                
                backpropagate(anInput, anLabel);
            }
            
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
            
            time ++;
        }
        return true;
    }
    
    //backpropagate one set in a batch
    bool backpropagate(float * inputs, float * labels){
        infer(inputs);
        
        //update output layer weights
        for (int i=0; i<m_numOutput; i++) {
            float dOut_dnet = cost_derivate(m_outputs[i] , labels[i], MeanSquared) * derivate(m_outputs[i], outputActivation);
            
            for (int j=0; j<m_numHidden; j++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_outputWeightGradients[outWeightIndex]  += dOut_dnet * m_hiddenOuputs[i];
            }
        }
        
        //calculate hidden layer gradient
        for (int j=0; j<m_numHidden; j++) {
            for (int i=0; i<m_numOutput; i++) {
                int outWeightIndex = getWeightIndex(j, i, m_numOutput);
                m_hiddenErrorGradientSum[j]  += cost_derivate(m_outputs[i] , labels[i], MeanSquared) * derivate(m_outputs[i], outputActivation) * m_outputWeights[outWeightIndex];
            }
        }
        
        //update hidden layer weights
        for (int i=0; i<m_numHidden; i++) {
            for (int j=0; j<m_numInput; j++) {
                int hiddenWeightIndex = getWeightIndex(j, i, m_numHidden);
                m_hiddenWeightGradients[hiddenWeightIndex] += m_hiddenErrorGradientSum[i] * m_hiddenWeights[hiddenWeightIndex];
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
