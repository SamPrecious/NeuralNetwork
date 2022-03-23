package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;

public class MultiLayerPerceptron {
	
	
	private static NeuralNetwork neuralNetwork = new NeuralNetwork(); 
	
	public static void main(String[] args) {

		MultiLayerPerceptron predict = new MultiLayerPerceptron();

	    inputData allInputs = getInputs();

		predict(allInputs);
	}

	

	public static void predict(inputData allInputs) {
		
		//ArrayList<SimpleMatrix> testing = allInputs.getTraining();
		//testing.get(0).print();
		
		//This represents the layers in a network EXCLUDING the input layer (as the input layer isnt treated the same as other layers with weights
		int networkSize = 3;
		
		double stepSize = 0.1;

		
		ArrayList<SimpleMatrix> trainingData = allInputs.getTraining();
		ArrayList<SimpleMatrix> trainingExpected = allInputs.getTrainingExpected();
		
		ArrayList<SimpleMatrix> validationData = allInputs.getValidation();
		ArrayList<SimpleMatrix> validationExpected = allInputs.getValidationExpected();
		
		ArrayList<SimpleMatrix> testingData = allInputs.getTesting();
		ArrayList<SimpleMatrix> testingExpected = allInputs.getTestingExpected();
				
		

		double[][] inputArray = { { 1, 0 } };

		// SimpleMatrix inputMatrix = new SimpleMatrix(inputArray);

		SimpleMatrix inputMatrix = new SimpleMatrix(inputArray); // An array containing all inputs as matrices
		
		
		/*
		 * inputData currentInputs = new inputData();
		 * 
		 * ArrayList<SimpleMatrix> testing = currentInputs.getTraining();
		 * 
		 * for(SimpleMatrix test: testing) { test.print(); }
		 */

		// Define all Matrices
		double[][] weightArray1 = { { 3, 6 }, { 4, 5 } };
		double[][] biasArray1 = { { 1, -6 } };
		double[][] weightArray2 = { { 2 }, { 4 } };
		double[][] biasArray2 = { { -3.92 } };
		
		double[][] weightArray3 = { { 3, 6 }, { 4, 5 } };
		double[][] biasArray3 = { { 1, -6 } };
		double[][] weightArray4 = { { 2 }, { 4 } };
		double[][] biasArray4 = { { -3.92 } };
		
		
		
		//Creates matrices for the weights and biases (using the arrays)
		neuralNetwork.addBiases(new SimpleMatrix(biasArray1));
		neuralNetwork.addBiases(new SimpleMatrix(biasArray2));
		
		neuralNetwork.addWeights(new SimpleMatrix(weightArray1));
		neuralNetwork.addWeights(new SimpleMatrix(weightArray2));
		
		
		neuralNetwork.addBiases(new SimpleMatrix(biasArray3));
		neuralNetwork.addBiases(new SimpleMatrix(biasArray3));
		
		neuralNetwork.addWeights(new SimpleMatrix(weightArray4));
		neuralNetwork.addWeights(new SimpleMatrix(weightArray4));
		
		
		
		SimpleMatrix hiddenNode = new SimpleMatrix(inputMatrix.numRows(), networkSize);
		SimpleMatrix hiddenNode2 = new SimpleMatrix(hiddenNode.numRows(), networkSize);
		SimpleMatrix outputNode = new SimpleMatrix(hiddenNode.numRows(), networkSize);
		
		ArrayList<SimpleMatrix> allLayers = new ArrayList<SimpleMatrix>();
		allLayers.add(inputMatrix);
		allLayers.add(hiddenNode);
		allLayers.add(hiddenNode2);
		allLayers.add(outputNode);
		

		double correctOutput = 1;
			
        HashMap<Integer, SimpleMatrix> deltaMatrices = new HashMap<Integer, SimpleMatrix>();  //Contains all delta values

        
        
       
        

        
        allLayers = forwardPass(allLayers, neuralNetwork);  
        
        
        
        
		allLayers.get(networkSize).print(); //output of forward pass

		
		//This forloop handles the backwards and forwards propagation. The number i is compared to represents the amount of epochs we use
		
		/*
		for (int i = 0; i < 20000; i++) {												
			//In this for loop we calculate the deltas for each layer - We use start from 1 as we dont calculate deltas for the input layer
			for(int j = networkSize; j>=1; j--) {
				//We have to calculate the delta of the output node differently
				if(j == networkSize) {
					double[][] currentDeltaArray = { { (correctOutput - allLayers.get(j).get(0, 0)) * (allLayers.get(j).get(0, 0) * (1 - allLayers.get(j).get(0, 0))) } };
		        	deltaMatrices.put(j, new SimpleMatrix(currentDeltaArray));
				}else {
					SimpleMatrix currentLayerDifferential = firstDifferential(allLayers.get(j).copy()); //We get a copy as to not modify the original structure
					deltaMatrices.put(j, deltaVal(currentLayerDifferential, neuralNetwork.getWeights(j), deltaMatrices.get(networkSize)));
				}
	        	
	        }
					
			
			/*
			 *  Updates the weight and biases of the neural network based on the delta values worked out
			 *  New weight: Old Weight + (Step Size * currentDelta * input)
			 *  New bias: Old Bias + (Step Size * currentDelta)
			 *//*

			neuralNetwork = updateValues(networkSize, neuralNetwork, allLayers, stepSize, deltaMatrices);
			
			
															
	        //After doing back propagation above, the program re-does the forward pass with updated values
			allLayers = forwardPass(allLayers, neuralNetwork);

	        //Gives us the output of the network for our current run
			allLayers.get(networkSize).print();
			
			
			
			
		}*/
		
		
		

	}
	
	private static ArrayList<SimpleMatrix> forwardPass(ArrayList<SimpleMatrix> layers, NeuralNetwork neuralNetwork) {
		
		
		//layers.get(0).print();	
		//layers.get(1).print();	
		//layers.get(2).print();	
		//layers.get(3).print();	

		neuralNetwork.getBiases(0).print();
		neuralNetwork.getBiases(1).print();
		neuralNetwork.getBiases(2).print();
		neuralNetwork.getBiases(3).print();

		
		//Struggling on 2nd hidden to output
		
		//Start from 1 because we dont perform operations on the input layer.
		for(int i = 1; i<layers.size(); i++) {
			layers.set(i, sigmoids(layers.get(i-1).mult(neuralNetwork.getWeights(i-1)).plus(neuralNetwork.getBiases(i-1)))); //Range of i-1 for Weights and Biases
			System.out.println("Success");
			//layers.set(i, sigmoids(layers.get(i-1));
		}
		
		return layers;
	}

	
	public static NeuralNetwork updateValues(int networkSize, NeuralNetwork neuralNetwork, ArrayList<SimpleMatrix> allLayers, double stepSize, HashMap<Integer, SimpleMatrix> deltaMatrices) {
		
		
		for(int i = 0; i<networkSize; i++) {
			
			deltaMatrices.get(i+1).transpose().mult(allLayers.get(i));
			neuralNetwork.setWeights(i, neuralNetwork.getWeights(i).plus(deltaMatrices.get(i+1).transpose().mult(allLayers.get(i)).scale(stepSize).transpose()));
 			neuralNetwork.setBiases(i, neuralNetwork.getBiases(i).plus(deltaMatrices.get(i+1).scale(stepSize)));
		}
		
		return neuralNetwork;

		//return weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());

	}
	
	
	public static SimpleMatrix updateBias(SimpleMatrix bias, double stepSize, SimpleMatrix delta) {
		// We want to get oldVal = oldVal + (stepSize*delta)
		//bias.plus(delta.scale(stepSize)).print();
		
		return bias.plus(delta.scale(stepSize));
	}

	public static SimpleMatrix updateWeight(SimpleMatrix weight, double stepSize, SimpleMatrix delta,
			SimpleMatrix input) {
		/*
		 * We want to get oldVal = oldVal + (stepSize*delta*input) This matrix
		 * calculation works out these values
		 */
		//input.print();
				
		return weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());

	}
	
	
	
	
	
	
	private static SimpleMatrix sigmoids(SimpleMatrix m) {
		DMatrixIterator it = m.iterator(false, 0, 0, m.numRows() - 1, m.numCols() - 1);

		// Iterates through the matrix and applies sigmoid
		while (it.hasNext()) {
			it.set(1 / (1 + Math.exp(-it.next())));
		}
		
		return m;
	}
	
	
	
	
	
	
	
	
	// Performs the forward through layer in a network
	private static SimpleMatrix forwardPass(SimpleMatrix currentLayer, SimpleMatrix input, SimpleMatrix weight, SimpleMatrix bias) {

		currentLayer = input.mult(weight).plus(bias);
		
		
		sigmoid(currentLayer);
		return currentLayer;
	}
	
	
	
		
		
	
	
	
	
	// Performs the forward through layer in a network
		
	
	

	

	// This next line multiplies the first differential with the (Weight * delta on
	// previous layer)

	public static SimpleMatrix deltaVal(SimpleMatrix currentDifferentialMatrix, SimpleMatrix weight,
			SimpleMatrix lastDifferentialMatrix) {

		return currentDifferentialMatrix.elementMult(weight.mult(lastDifferentialMatrix).transpose());
	}

	private static SimpleMatrix firstDifferential(SimpleMatrix currentLayer) {
		DMatrixIterator it = currentLayer.iterator(false, 0, 0, currentLayer.numRows() - 1, currentLayer.numCols() - 1);

		// Iterates through the matrix and applies sigmoid
		while (it.hasNext()) {
			double curVal = it.next();
			it.set(curVal * (1 - curVal));
		}

		return currentLayer;
	}

	

	private static void sigmoid(SimpleMatrix m) {

		DMatrixIterator it = m.iterator(false, 0, 0, m.numRows() - 1, m.numCols() - 1);

		// Iterates through the matrix and applies sigmoid
		while (it.hasNext()) {
			it.set(1 / (1 + Math.exp(-it.next())));
		}

	}
	
	/*
	 * This file reads the datasetReadable xlsx file
	 * All the data read then gets sent to the inputData object, containing 3 arrays
	 * These 3 arrays are Training, Validation and Testing, which are split 60%, 20%, 20% respectively
	 */
	public static inputData getInputs() {

		inputData allInputs = new inputData();

		// allInputs.addTesting();

		try {

			FileInputStream excelFile = new FileInputStream(new File("datasetReadable.xlsx"));
			Workbook workbook = new XSSFWorkbook(excelFile);
			Sheet datatypeSheet = workbook.getSheetAt(0);
			Iterator<Row> iterator = datatypeSheet.iterator();

			int width = 0;
			Row checkWidth = iterator.next(); // Uses the header row to check the width of the excel document
			Iterator<Cell> cellIteratorWidth = checkWidth.iterator();

			while (cellIteratorWidth.hasNext()) {
				cellIteratorWidth.next();
				width++;
			}
			
			int datasetDeterminant = 1; //This integer determines what dataset the data will fall into (split 60/20/20 so 3:1:1)

			while (iterator.hasNext()) {
				// System.out.println(currentRowIndex);

				Row currentRow = iterator.next();
				Iterator<Cell> cellIterator = currentRow.iterator();

				double[][] currentRowArray = new double[1][width - 1];
				double[][] currentRowPredictedArray = new double[1][1]; // 1x1 array for the output

				int i = 0;

				while (cellIterator.hasNext()) {

					Cell currentCell = cellIterator.next();

					if (i == width - 1) {
						currentRowPredictedArray[0][0] = currentCell.getNumericCellValue();
					} else {
						currentRowArray[0][i] = currentCell.getNumericCellValue();
					}

					i++;

				}

				SimpleMatrix currentRowMatrix = new SimpleMatrix(currentRowArray);
				SimpleMatrix currentRowPredictedMatrix = new SimpleMatrix(currentRowPredictedArray);

				
				if(datasetDeterminant<=3) { //First 3 variables go training
					allInputs.addTraining(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
				}else if(datasetDeterminant<=4) { //Next row for validation
					allInputs.addValidation(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
				}else { //Final row for testing data, then we reset the index to repeat the spread
					allInputs.addTesting(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant = 1;
				}
				
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return allInputs;
		
		
	}

}
