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
		 * 		Store all weights coming in as osne matrix
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
			    //hiddenNode.mult(weightMatrix2).plus(biasMatrix2)
			    
			    
			    hiddenNode = forwardPass(hiddenNode, inputMatrix, weightMatrix1, biasMatrix1);
			    outputNode = forwardPass(outputNode, hiddenNode, weightMatrix2, biasMatrix2);
			    
						    
			    outputNode.print();
	}
	
	//Performs the forward pass in a network
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
