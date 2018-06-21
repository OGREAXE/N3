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
    float * m_outputs;
    
    ActivationFunction outputActivation;
    
    CostFunction costFunction;
    
    //backpropagate
    float * m_hiddenOuputErrorGradients;
    float * m_outputErrorGradients;
    
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
    
    bool backpropagate(float * labelInputs, float * labels, int numLabels){
        infer(labelInputs);
        for (int i=0; i<m_numOutput; i++) {
            float dOut = cost_derivate(m_outputs , labelInputs, MeanSquared);
            float dOut_dnet = dOut * derivate(m_outputs[i], outputActivation);
            m_outputErrorGradients[i] = dOut_dnet;
            for (int j=0; j<m_numHidden; j++) {
                float dHiddenWeight= dOut_dnet * m_hiddenOuputs[i];
            }
        }
        
        for (int j=0; j<m_numHidden; j++) {
            m_hiddenOuputErrorGradients[j] = 0;
            for (int i=0; i<m_numOutput; i++) {
                m_hiddenOuputErrorGradients[j] += m_outputErrorGradients[i] * m_outputWeights[i + j * m_numOutput];
            }
            
        }
        
        for (int i=0; i<m_numInput; i++) {
            for (int j=0; j<m_numHidden; j++){
                float dinputWeight = m_hiddenOuputErrorGradients[j] * derivate(m_hiddenOuputs[j], hiddenActivation) * m_inputCache[i];
            }
        }
        
        return true;
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
    
    float cost_derivate(float * output, float * target, CostFunction costFunction){
        return 0;
    }
};

#endif /* NextNeuralNet_hpp */
