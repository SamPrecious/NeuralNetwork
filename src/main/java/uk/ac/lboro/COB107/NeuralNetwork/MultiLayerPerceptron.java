package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.Arrays;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class MultiLayerPerceptron {
		
	public static void main(String[] args) {
		/*
		 * How to store:
		 * 		This is brief as im not yet certain however I think
		 * 		The left matrix changes a lot, so we probably shouldnt store this with a neuron
		 * 		The right matrix should be stored within a neuron
		 * 		Could store left matrix 
		 * ACTUALLY:
		 * 		Could store each layer. I think this makes more sense.
		 * 		Store all weights coming in as one matrix
		 * 		Store all biases as another 1D matrix
		 * 		Store 
		 */		
		MultiLayerPerceptron predict = new MultiLayerPerceptron();
		predict();
		
		
	    
	}
	
	
	public static void predict() {
				//This goes over the worked example in the lecture: 
				double[][] inputArray = {{1,0}};
			    SimpleMatrix inputMatrix = new SimpleMatrix(inputArray);

			    //Define all Matrices
			    double[][] weightArray1 =  {{3, 6},{4,5}};
			    SimpleMatrix weightMatrix1 = new SimpleMatrix(weightArray1);
			    
			    double[][] biasArray1 = {{1,-6}};
			    SimpleMatrix biasMatrix1 = new SimpleMatrix(biasArray1);
			    
			    double[][] weightArray2 = {{2},{4}};
			    SimpleMatrix weightMatrix2 = new SimpleMatrix(weightArray2);
			    
			    double[][] biasArray2 = {{-3.92}};
			    SimpleMatrix biasMatrix2 = new SimpleMatrix(biasArray2);
			    
			    SimpleMatrix hiddenNode = new SimpleMatrix(inputMatrix.numRows(),weightMatrix1.numCols());
			    SimpleMatrix outputNode = new SimpleMatrix(hiddenNode.numRows(), weightMatrix2.numCols());
			    
			   

			    
			    double outputDelta;
			    double[] hiddenDelta = new double[outputNode.numCols()];
			    
			    double stepSize = 0.1;
			    
			    double correctOutput = 1;

			    
			    
			    
			    
			    hiddenNode = forwardPass(hiddenNode, inputMatrix, weightMatrix1, biasMatrix1);
			    
			    

			    
			    outputNode = forwardPass(outputNode, hiddenNode, weightMatrix2, biasMatrix2);

			    
			    //The rest is a backward pass, we need most values in this function, so there is no point creating another function for this.
			    
			    outputDelta  = (correctOutput - outputNode.get(0, 0))* (outputNode.get(0, 0) * (1-outputNode.get(0, 0)));

			    double[][] outputDeltaArray = {{outputDelta}};
			    SimpleMatrix outputDeltaMatrix = new SimpleMatrix(outputDeltaArray);
			    
			    			    			    			    
			    //Here we clone our hidden node matrix so we can get a matrix of current differentials
			    SimpleMatrix hiddenDeltaMatrix = hiddenNode.copy();
			    firstDifferential(hiddenDeltaMatrix);
			    hiddenDeltaMatrix = deltaVal(hiddenDeltaMatrix, weightMatrix2, outputDeltaMatrix);		    		    		    		    			    			    		    
			    
			    //New weight: Old Weight + (Step Size * currentDelta * input)			    

			    
			    weightMatrix1 = updateWeight(weightMatrix1, stepSize, hiddenDeltaMatrix, inputMatrix);
			    biasMatrix1 = updateBias(biasMatrix1, stepSize, hiddenDeltaMatrix);
			    weightMatrix2 = updateWeight(weightMatrix2, stepSize, outputDeltaMatrix, hiddenNode);
			    biasMatrix2 = updateBias(biasMatrix2, stepSize, outputDeltaMatrix);
			    
			    
	}
	
	
	
	public static SimpleMatrix updateBias(SimpleMatrix bias, double stepSize, SimpleMatrix delta) {
		//We want to get oldVal = oldVal + (stepSize*delta)
		return bias.plus(delta.scale(stepSize));
	}
	
	public static SimpleMatrix updateWeight(SimpleMatrix weight, double stepSize, SimpleMatrix delta, SimpleMatrix input) {						
		/*
		 * We want to get oldVal = oldVal + (stepSize*delta*input)
		 * This matrix calculation works out these values
		 */

		return weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());
						
		
	}
	
	
	// This next line multiplies the first differential with the (Weight * delta on previous layer)
	
	public static SimpleMatrix deltaVal(SimpleMatrix currentDifferentialMatrix, SimpleMatrix weight, SimpleMatrix lastDifferentialMatrix) {
		
		return currentDifferentialMatrix.elementMult(weight.mult(lastDifferentialMatrix).transpose());
	}
	
	
	
	private static SimpleMatrix firstDifferential(SimpleMatrix currentLayer) {
		DMatrixIterator it = currentLayer.iterator(false, 0, 0, currentLayer.numRows()-1, currentLayer.numCols()-1);
	    
	    //Iterates through the matrix and applies sigmoid
	    while(it.hasNext()) {
	    	double curVal = it.next();
	    	it.set(curVal * (1-curVal)); 
	    }
		
		return currentLayer;
	}
					
	
	//Performs the forward through layer in a network
	private static SimpleMatrix forwardPass(SimpleMatrix currentLayer, SimpleMatrix input, SimpleMatrix weight, SimpleMatrix bias) {
				
		currentLayer = input.mult(weight).plus(bias);
				
		sigmoid(currentLayer);
		
		return currentLayer;
	}

	private static void sigmoid(SimpleMatrix m) {
		
	    DMatrixIterator it = m.iterator(false, 0, 0, m.numRows()-1, m.numCols()-1);
	    
	    //Iterates through the matrix and applies sigmoid
	    while(it.hasNext()) {
	    	it.set(1/(1+Math.exp(-it.next()))); 
	    }
	    
	}
	
	
	
	
}
