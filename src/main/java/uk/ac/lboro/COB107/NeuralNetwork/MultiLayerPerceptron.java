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
import java.util.Random;

public class MultiLayerPerceptron {

	private static NeuralNetwork neuralNetwork = new NeuralNetwork();

	public static void main(String[] args) {

		MultiLayerPerceptron predict = new MultiLayerPerceptron();

		inputData allInputs = getInputs();

		predict(allInputs);
	}

	
	/*
	 * NOTE: Columns are the variables inside the curly brackets, rows are curly brackets
	 * Guide to adding a new layer:
	 * 		For each layer, the previous weight node must have one more column
	 * 		New Layer:
	 * 			Weight going in must have the amount of columns for each node current layer
	 * 			Weight going in must have the amount of rows for each node on previous layer
	 * 	Say for example we want to increase the size of the input node.
	 * 	We have now +1ed the amount of inputs being used, this means the output weight array will have the same amount of columns, but +1 row
	 * 		
	 * 
	 */
	public static void predict(inputData allInputs) {

		
		ArrayList<Integer> layerSizes = new ArrayList<Integer>();
		
		layerSizes.add(2); //Input Node
		layerSizes.add(3); //Hidden Nodes
		layerSizes.add(2); //Hidden Nodes
		layerSizes.add(1); //Output
		

		double rangeMin = 10;
		double rangeMax = 20;
		
		Random r = new Random();
		double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
		
		//System.out.println(randomValue);
		
		//Inbetween layer xi and xi+1, the weights must have xi (node amount) rows, and xi+1 (node amount) columns, so this calculates that,
		int rows = layerSizes.get(0);
		for(int i = 1; i<layerSizes.size(); i++) {
			int column = layerSizes.get(i);
			System.out.println("Rows: "+ rows);
			System.out.println("Columns: "+ column + "");
			rows = column;
		}
		
		
		
		
		// ArrayList<SimpleMatrix> testing = allInputs.getTraining();
		// testing.get(0).print();

		// This represents the layers in a network EXCLUDING the input layer (as the
		// input layer isnt treated the same as other layers with weights
		int networkSize = 2;

		double stepSize = 0.1;

		ArrayList<SimpleMatrix> trainingData = allInputs.getTraining();
		ArrayList<SimpleMatrix> trainingExpected = allInputs.getTrainingExpected();

		ArrayList<SimpleMatrix> validationData = allInputs.getValidation();
		ArrayList<SimpleMatrix> validationExpected = allInputs.getValidationExpected();

		ArrayList<SimpleMatrix> testingData = allInputs.getTesting();
		ArrayList<SimpleMatrix> testingExpected = allInputs.getTestingExpected();

		
		double[] trainingTotalMax = allInputs.getTrainingTotalMax();
		double[] trainingTotalMin = allInputs.getTrainingTotalMin();
		double[] trainingPredictandMax = allInputs.getTrainingPredictandMax();
		double[] trainingPredictandMin = allInputs.getTrainingPredictandMin();
		
		
		double[] validationTotalMax = allInputs.getValidationTotalMax();
		double[] validationTotalMin = allInputs.getValidationTotalMin();
		double[] validationPredictandMax = allInputs.getValidationPredictandMax();
		double[] validationPredictandMin = allInputs.getValidationPredictandMin();
		
		
		double[][] inputArray = { { 1, 0 }};
		
		
		
		
		// SimpleMatrix inputMatrix = new SimpleMatrix(inputArray);

		SimpleMatrix inputMatrix = new SimpleMatrix(inputArray); // An array containing all inputs as matrices
		
		//Inputs DONT change, we can change them from an array to an arraylist, then just compare extractions of the row instead of the whole list. CHANGE THIS... OR CHANGE THE STANDARDISATION METHOD TO LOOP THROUGH ARRAYLIST INSTEAD OF MATRICES
		
		
		trainingData.get(0).print();
		//trainingData = standardise(trainingData, trainingTotalMax, trainingTotalMin);
		
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
		//double[][] weightArray2 = { { 3, 6, 2 }, { 4, 5, 2 } }; 
		//double[][] biasArray2 = { { 1, -6, 4} };
		double[][] weightArray2 = { { 2 }, { 4 } };
		double[][] biasArray2 = { { -3.92 } };

		
		//To add layer, new layer must have amount of inputs as previous layer (i.e. prev layer = 2 inputs, current layer could have a weight array of e.g. {3,4,2}{3,2,1} <- Assumes our new hidden layer has 3 nodes. Our next layer (so w2 in this case) must get 1 more weight added (so another {})
		//to increase size of node: increasing node size means you must: increase the amount of weights going in, i.e. w3 +1 per node per previous array. Increase biases by +1 per node
		

		// Creates matrices for the weights and biases (using the arrays)
		neuralNetwork.addBiases(new SimpleMatrix(biasArray1));
		neuralNetwork.addBiases(new SimpleMatrix(biasArray2));
		//neuralNetwork.addBiases(new SimpleMatrix(biasArray3));

		neuralNetwork.addWeights(new SimpleMatrix(weightArray1));
		neuralNetwork.addWeights(new SimpleMatrix(weightArray2));
		//neuralNetwork.addWeights(new SimpleMatrix(weightArray3));


		SimpleMatrix hiddenNode = new SimpleMatrix(inputMatrix.numRows(), networkSize);

		SimpleMatrix hiddenNode2 = new SimpleMatrix(hiddenNode.numRows(), networkSize);

		SimpleMatrix outputNode = new SimpleMatrix(hiddenNode2.numRows(), networkSize);

		ArrayList<SimpleMatrix> allLayers = new ArrayList<SimpleMatrix>();
		allLayers.add(inputMatrix);
		allLayers.add(hiddenNode);
		//allLayers.add(hiddenNode2);
		allLayers.add(outputNode);
		
		
		
		
				
		
		double correctOutput = 0.9;

		HashMap<Integer, SimpleMatrix> deltaMatrices = new HashMap<Integer, SimpleMatrix>(); // Contains all delta
																								// values


		/*
		 * neuralNetwork.getWeights(0).print(); neuralNetwork.getWeights(1).print();
		 * neuralNetwork.getWeights(2).print();
		 */


		// This forloop handles the backwards and forwards propagation. The number i is
		// compared to represents the amount of epochs we use
		
		
		//Every time allLayers.get(0) is referenced, replace with currentInput
		
		for (int i = 0; i < 2000; i++) {
			
			for(int x = 0; x<inputMatrix.numCols(); x++) {
				SimpleMatrix currentInput = inputMatrix.extractVector(false, x);
			}
			
			allLayers = forwardPass(allLayers, neuralNetwork);
			//allLayers.get(networkSize).print();

			
			// In this for loop we calculate the deltas for each layer - We use start from 1
			// as we dont calculate deltas for the input layer
			for (int j = networkSize; j >= 1; j--) {
				// We have to calculate the delta of the output node differently
				if (j == networkSize) {
					double[][] currentDeltaArray = { { (correctOutput - allLayers.get(j).get(0, 0))
							* (allLayers.get(j).get(0, 0) * (1 - allLayers.get(j).get(0, 0))) } };
					deltaMatrices.put(j, new SimpleMatrix(currentDeltaArray));
				} else {

					SimpleMatrix currentLayerDifferential = firstDifferential(allLayers.get(j).copy()); // We get a copy
																										// as to not
																										// modify the
																										// original
																										// structure

					deltaMatrices.put(j, deltaVal(currentLayerDifferential, neuralNetwork.getWeights(j),
							deltaMatrices.get(networkSize))); // Calculating based on the final delta matrix, we want to
																// calculate the previous one

				}

			}

			/*
			 * Updates the weight and biases of the neural network based on the delta values
			 * worked out New weight: Old Weight + (Step Size * currentDelta * input) New
			 * bias: Old Bias + (Step Size * currentDelta)
			 */

			neuralNetwork = updateValues(networkSize, neuralNetwork, allLayers, stepSize, deltaMatrices);

			// After doing back propagation above, the program re-does the forward pass with
			// updated values

			// Gives us the output of the network for our current run

		}
		allLayers = forwardPass(allLayers, neuralNetwork);
		
		SimpleMatrix output = allLayers.get(networkSize);
		output = destandardise(output, 1, 0);
		output.print();


	}

	private static ArrayList<SimpleMatrix> forwardPass(ArrayList<SimpleMatrix> layers, NeuralNetwork neuralNetwork) {

		// Start from 1 because we don't perform operations on the input layer.
		for (int i = 1; i < layers.size(); i++) {
			layers.set(i, sigmoids(
					layers.get(i - 1).mult(neuralNetwork.getWeights(i - 1)).plus(neuralNetwork.getBiases(i - 1)))); // Range of i-1 for Weights and Biases
		}

		return layers;
	}
	
	//Standardises all data to the range [0.1, 0.9]
	public static SimpleMatrix standardise(SimpleMatrix input, double[] max, double[] min) {
		
		/*
		double[] newMax = {12,32,36,34,9};
		double[] newMin = {1,1,1,8,4};
		
		double[][] newInputs = {{12,32,1,34,5},{1,3,12,13,4},{7,1,36,8,9}};
		SimpleMatrix inMatrix = new SimpleMatrix(newInputs);*/
		
		for(int i = 0; i<input.numCols(); i++) {
			DMatrixIterator it = input.iterator(false, 0, i, input.numRows() - 1, i);  //Iterate through current column
			while (it.hasNext()) {
				it.set((0.8 * (it.next()-min[i])/(max[i]-min[i])) + 0.1);
				
			}
		}
		
	
		return input;
	}
	
	//Un-standardises the matrix for the correct outputs
	public static SimpleMatrix destandardise(SimpleMatrix input, int max, int min) {				
		DMatrixIterator it = input.iterator(false, 0, 0, input.numRows() - 1, input.numCols() - 1);

		while (it.hasNext()) {
			it.set(((it.next() - 0.1)/0.8)*(max - min) + min) ;
		}
		return input;
	}
	
	
	
	
	public static NeuralNetwork updateValues(int networkSize, NeuralNetwork neuralNetwork,
			ArrayList<SimpleMatrix> allLayers, double stepSize, HashMap<Integer, SimpleMatrix> deltaMatrices) {

		for (int i = 0; i < networkSize; i++) {

			deltaMatrices.get(i + 1).transpose().mult(allLayers.get(i));
			neuralNetwork.setWeights(i, neuralNetwork.getWeights(i)
					.plus(deltaMatrices.get(i + 1).transpose().mult(allLayers.get(i)).scale(stepSize).transpose()));
			neuralNetwork.setBiases(i, neuralNetwork.getBiases(i).plus(deltaMatrices.get(i + 1).scale(stepSize)));
		}

		return neuralNetwork;

		// return
		// weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());

	}

	public static SimpleMatrix updateBias(SimpleMatrix bias, double stepSize, SimpleMatrix delta) {
		return bias.plus(delta.scale(stepSize));
	}

	public static SimpleMatrix updateWeight(SimpleMatrix weight, double stepSize, SimpleMatrix delta,
			SimpleMatrix input) {
		/*
		 * We want to get oldVal = oldVal + (stepSize*delta*input) This matrix
		 * calculation works out these values
		 */
		// input.print();

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

	// This next line multiplies the first differential with the (Weight * delta on
	// previous layer)

	public static SimpleMatrix deltaVal(SimpleMatrix currentDifferentialMatrix, SimpleMatrix weights, SimpleMatrix lastDifferentialMatrix) {

		SimpleMatrix averageDeltaWeight = new SimpleMatrix(weights.numRows(), lastDifferentialMatrix.numCols());

		for (int i = 0; i < weights.numCols(); i++) {
			// Gets each value of the weight matrix, and adds it up
			averageDeltaWeight = averageDeltaWeight.plus(weights.extractVector(false, i).mult(lastDifferentialMatrix));

		}
		double numCols = weights.numCols();
		averageDeltaWeight = averageDeltaWeight.scale(1 / numCols); // This gets the average weights, i.e. if we had 2

		return currentDifferentialMatrix.elementMult(averageDeltaWeight.transpose());
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

	/*
	 * This file reads the datasetReadable xlsx file All the data read then gets
	 * sent to the inputData object, containing 3 arrays These 3 arrays are
	 * Training, Validation and Testing, which are split 60%, 20%, 20% respectively
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

			int datasetDeterminant = 1; // This integer determines what dataset the data will fall into (split 60/20/20 so 3:1:1)

			
			double[] totalRowMaxValidation = new double[width-1];
			double[] totalRowMinValidation = new double[width-1];
			double[] predictandMaxValidation = new double[1];
			double[] predictandMinValidation = new double[1];
			predictandMinValidation[0] = Double.MAX_VALUE;						
			
			double[] totalRowMaxTraining = new double[width-1];
			double[] totalRowMinTraining = new double[width-1];
			double[] predictandMaxTraining = new double[1];
			double[] predictandMinTraining = new double[1];
			predictandMinTraining[0] = Double.MAX_VALUE;
			
			for(int i = 0; i<totalRowMinValidation.length; i++)  { //Makes all variables in the predictand min column the max value, so we dont accidentally put a value lower than minimum in here
				totalRowMinValidation[i] = Double.MAX_VALUE;
			}
			for(int i = 0; i<totalRowMinTraining.length; i++) { //Makes all variables in the predictand min column the max value, so we dont accidentally put a value lower than minimum in here
				totalRowMinTraining[i] = Double.MAX_VALUE;
			}
			
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
						double currentVal = currentCell.getNumericCellValue();
						currentRowPredictedArray[0][0] = currentVal;
						
						
						if (datasetDeterminant <= 3) { // First 3 variables go training
							predictandMaxTraining[0] = Math.max(predictandMaxTraining[0], currentVal);  //Compares last value to current to check for max
							predictandMinTraining[0] = Math.min(predictandMinTraining[0], currentVal);  //Compares last value to current to check for max
						}else if (datasetDeterminant <= 4) { // First 3 variables go training
							predictandMaxValidation[0] = Math.max(predictandMaxValidation[0], currentVal);  //Compares last value to current to check for max
							predictandMinValidation[0] = Math.min(predictandMinValidation[0], currentVal);  //Compares last value to current to check for max
						}
						
					} else {
						double currentVal = currentCell.getNumericCellValue();
						currentRowArray[0][i] = currentVal;
						
						
						if (datasetDeterminant <= 3) { // First 3 variables go training
							totalRowMaxTraining[i] = Math.max(totalRowMaxTraining[i], currentVal);  //Compares last value to current to check for max
							totalRowMinTraining[i] = Math.min(totalRowMinTraining[i], currentVal);  //Compares last value to current to check for max
						}else if (datasetDeterminant <= 4) { // First 3 variables go training
							totalRowMaxValidation[i] = Math.max(totalRowMaxValidation[i], currentVal);  //Compares last value to current to check for max
							totalRowMinValidation[i] = Math.min(totalRowMinValidation[i], currentVal);  //Compares last value to current to check for max
						}
						
					}
					
					i++;

				}
				
				
				allInputs.addMaxMins(totalRowMaxTraining, totalRowMinTraining, predictandMaxTraining, predictandMinTraining, totalRowMaxValidation, totalRowMinValidation, predictandMaxValidation, predictandMinValidation);

				SimpleMatrix currentRowMatrix = new SimpleMatrix(currentRowArray);
				SimpleMatrix currentRowPredictedMatrix = new SimpleMatrix(currentRowPredictedArray);

				if (datasetDeterminant <= 3) { // First 3 variables go training
					allInputs.addTraining(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
				} else if (datasetDeterminant <= 4) { // Next row for validation
					allInputs.addValidation(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
				} else { // Final row for testing data, then we reset the index to repeat the spread
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
