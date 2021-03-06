package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;
import java.util.HashMap;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;

public class MultiLayerPerceptron {

	private static NeuralNetwork neuralNetwork = new NeuralNetwork();

	public static void main(String[] args) {

		//Reads the excel file, after it has been put in random order
		inputData allInputs = getInputs();

		predict(allInputs);
	}


	@SuppressWarnings("unchecked")
	public static void predict(inputData allInputs) {
		
		Scanner scan = new Scanner(System.in);
		System.out.println("How many hidden layers do we want? ");
		int hiddenLayerSize = scan.nextInt();
		
		ArrayList<Integer> layerSizes = new ArrayList<Integer>();

		
		int networkSize = hiddenLayerSize+1; //This is the amount of layers excluding the input layer (so +1 for output)

		double stepSize = 0.3; //Learning Rate

		ArrayList<SimpleMatrix> trainingData = allInputs.getTraining();
		ArrayList<SimpleMatrix> trainingExpected = allInputs.getTrainingExpected();
			
		double[] trainingTotalMax = allInputs.getTrainingTotalMax();
		double[] trainingTotalMin = allInputs.getTrainingTotalMin();
		double[] trainingPredictandMax = allInputs.getTrainingPredictandMax();
		double[] trainingPredictandMin = allInputs.getTrainingPredictandMin();

		

		SimpleMatrix inputMatrix = trainingData.get(0); // An array containing all inputs as matrices
		
		//We standardise our entire dataset to a range of   -2/2n -> 2/2n
		trainingData = standardise(trainingData, trainingTotalMax, trainingTotalMin);
		trainingExpected = standardise(trainingExpected, trainingPredictandMax, trainingPredictandMin);


		//allLayers contains the inputs going through each layer (like the input layer being the first)
		ArrayList<SimpleMatrix> allLayers = new ArrayList<SimpleMatrix>();
		allLayers.add(inputMatrix);
		layerSizes.add(inputMatrix.numCols()); 

		
		//This loop generates all layers with an appropriate size of rows and columns to match the ANN structure
		for (int i = 0; i < networkSize; i++) {
			SimpleMatrix currentLayer = new SimpleMatrix(allLayers.get(i).numRows(), 1); // 1 wide, and previous node of inputs long																		
			allLayers.add(currentLayer);
			if (i == networkSize - 1) { //We only use one output node for this model, so its only 1 big
				layerSizes.add(1);
			} else { // Change this to get user input, using base number 2 now as its way easier to
						// test without inputting each time
				System.out.println("How big do you want hidden layer "+ (i+1) +" to be?");
				int currentLayerSize = scan.nextInt();
				layerSizes.add(currentLayerSize);
			}
		}

		// Inbetween layer xi and xi+1, the weights must have xi (node amount) rows, and
		// xi+1 (node amount) columns, so this calculates that. We calculate biases for
		// xi+1 as well, we use the range -2/n -> 2/n, where n is input size
		int row = layerSizes.get(0);
		double inputs = trainingData.get(0).numCols();
		for (int i = 1; i < layerSizes.size(); i++) {
			int column = layerSizes.get(i);
			double[][] currentWeightArray = new double[row][column];

			for (int x = 0; x < currentWeightArray.length; x++) {
				for (int y = 0; y < currentWeightArray[x].length; y++) {
					//Generate random number within range
					double randomNum = ThreadLocalRandom.current().nextDouble(-(2 / inputs), 2 / inputs);
					currentWeightArray[x][y] = randomNum;
				}
			}

			
			double[][] currentBiasArray = new double[1][column];
			
			for (int x = 0; x < currentBiasArray[0].length; x++) {
				double randomNum = ThreadLocalRandom.current().nextDouble(-(2 / inputs), 2 / inputs); 
				currentBiasArray[0][x] = randomNum;
			}

			SimpleMatrix currentWeightMatrix = new SimpleMatrix(currentWeightArray);
			SimpleMatrix currentBiasMatrix = new SimpleMatrix(currentBiasArray);

			neuralNetwork.addWeights(currentWeightMatrix);
			neuralNetwork.addBiases(currentBiasMatrix);

			row = column;
		}

		// To add layer, new layer must have amount of inputs as previous layer (i.e.
		// prev layer = 2 inputs, current layer could have a weight array of e.g.
		// {3,4,2}{3,2,1} <- Assumes our new hidden layer has 3 nodes. Our next layer
		// (so w2 in this case) must get 1 more weight added (so another {})
		// to increase size of node: increasing node size means you must: increase the
		// amount of weights going in, i.e. w3 +1 per node per previous array. Increase
		// biases by +1 per node

		double correctOutput = trainingExpected.get(0).get(0, 0);

		HashMap<Integer, SimpleMatrix> deltaMatrices = new HashMap<Integer, SimpleMatrix>(); 

		XYSeries absoluteValues = new XYSeries("Absolute Values");
		XYSeries MSRESeries = new XYSeries("Mean Squared Relative Errors");

		// This forloop handles the backwards and forwards propagation. The number i is
		// compared to represents the amount of epochs we use

		// Every time allLayers.get(0) is referenced, replace with currentInput
		
		
		scan.close();

		
		int errorIncreasing = 0; //This is to check for overfitting with the Validation set
		double lastMSREVal = Double.MAX_VALUE;
		int epochs = 10000;
		
		long startTime = System.nanoTime();
		for (int i = 0; i < epochs; i++) {
			double currentAbsolute = 0;
			for (int x = 0; x <= allInputs.getTrainingSize(); x++) { 
				correctOutput = trainingExpected.get(x).get(0, 0);
				
				allLayers.set(0, trainingData.get(x));

				allLayers = forwardPass(allLayers, neuralNetwork);
								
				
				// Here we calculate deltas per layers, finish at 1 as we dont do input layer
				for (int j = networkSize; j >= 1; j--) {
					// System.out.println(j);
					// We have to calculate the delta of the output node differently
					if (j == networkSize) {
						double[][] currentDeltaArray = { { (correctOutput - allLayers.get(j).get(0, 0))
								* ((allLayers.get(j).get(0, 0) * (1 - allLayers.get(j).get(0, 0)))) } };  
								//1 - Math.pow((allLayers.get(j).get(0, 0), 2)) <- tanh
						deltaMatrices.put(j, new SimpleMatrix(currentDeltaArray));

					} else {
						// We get a copy as to not modify the original structure
						SimpleMatrix currentLayerDifferential = firstDifferential(allLayers.get(j).copy());
						
						deltaMatrices.put(j, deltaVal(currentLayerDifferential, neuralNetwork.getWeights(j),
								deltaMatrices.get(j+1))); 											

					}

				}

				/*
				 * Updates the weight and biases of the neural network based on the delta values
				 * worked out New weight: Old Weight + (Step Size * currentDelta * input) New
				 * bias: Old Bias + (Step Size * currentDelta)
				 */

				neuralNetwork = updateValues(i, x, networkSize, neuralNetwork, allLayers, stepSize, deltaMatrices);

				

			}
			neuralNetwork.setAllLayers(allLayers);
			//Save current weight changes to last weight changes before wiping current changes for next iteration
			neuralNetwork.setLastWeightChanges((ArrayList<ArrayList<SimpleMatrix>>) neuralNetwork.getCurrentWeightChanges().clone());
			neuralNetwork.clearCurrentWeightChanges();

			
			neuralNetwork.setLastBiasChanges((ArrayList<ArrayList<SimpleMatrix>>) neuralNetwork.getCurrentBiasChanges().clone());
			neuralNetwork.clearCurrentBiasChanges();
			
			
			//neuralNetwork.
			//Calculates error for this epoch
			double currentAbsoluteError = absoluteErrorCalc(allInputs, neuralNetwork, networkSize);
			
			
			Double currentAbsoluteCast = (Double) currentAbsoluteError;
			absoluteValues.add(i, currentAbsoluteCast);					
			
			if(i % 5 == 0) {
				
				double currentMSREVal = checkValidation(allInputs, neuralNetwork, networkSize);
				MSRESeries.add(i, currentMSREVal);
				if(currentMSREVal>lastMSREVal) { //Error is increasing 
					errorIncreasing++;
					if(errorIncreasing == 4) { //If the MSRE goes up much in the last few epochs, break early
						//System.out.println("Breaking");
						break;
					}
				}else {
					if(errorIncreasing != 0) {
						errorIncreasing--;
					}
				}
				lastMSREVal = currentMSREVal;

			}
			
		}		
		
		long endTime = System.nanoTime();
		
		System.out.println("The system took " + (endTime - startTime) / 1000000 +" milliseconds to complete");
		
	    XYSeriesCollection absoluteDataSet = new XYSeriesCollection();
	    absoluteDataSet.addSeries(absoluteValues);
		
		ChartPlotter absoluteChart = new ChartPlotter("Absolute vs Epochs", 
	    		"Absolute v Epochs",
	    		absoluteDataSet, "Epochs", "Absolute Error");
		absoluteChart.pack();
		RefineryUtilities.centerFrameOnScreen( absoluteChart );

	    absoluteChart.setVisible( true );
	    
	    
	    XYSeriesCollection msreDataSet = new XYSeriesCollection();
	    msreDataSet.addSeries(MSRESeries);
		
		ChartPlotter msreChart = new ChartPlotter("Mean Squared Relative Error", 
	    		"MSRE v Epochs",
	    		msreDataSet, "Epochs", "MSRE");
		msreChart.pack();
		RefineryUtilities.centerFrameOnScreen(msreChart);

		msreChart.setVisible( true );
		
	    neuralNetwork.setAllLayers(allLayers);
	    runTest(allInputs, neuralNetwork, networkSize);
	    
	}
	
	
	public static double absoluteErrorCalc(inputData inputs, NeuralNetwork neuralNetwork, int networkSize) {
		ArrayList<SimpleMatrix> trainingData = inputs.getTraining();
		ArrayList<SimpleMatrix> trainingExpected = inputs.getTrainingExpected();
		ArrayList<SimpleMatrix> allLayers = neuralNetwork.getAllLayers(); 
		
		double currentAbsolute = 0;

		for(int i = 0; i<=inputs.getTrainingSize(); i++) {
			allLayers.set(0, trainingData.get(i));			
			allLayers = forwardPass(allLayers, neuralNetwork);
			
			double correctOutput = trainingExpected.get(i).get(0, 0);
			//tempAbsolute is the absolute for this cycle
			double tempAbsolute = correctOutput - allLayers.get(networkSize).get(0, 0);
			
			if (tempAbsolute < 0) {
				tempAbsolute = -tempAbsolute;
			}
			
			
			currentAbsolute = currentAbsolute + tempAbsolute;
			
		}
		//Calculate average absolute error for this epoch
		currentAbsolute = currentAbsolute/inputs.getTrainingSize();
		
		return currentAbsolute;
		
	}
	
	//Here we check the validation set and return the MSRE for this iteration
	public static double checkValidation(inputData inputs, NeuralNetwork neuralNetwork, int networkSize) {
		
		ArrayList<SimpleMatrix> validationData = inputs.getValidation();
		ArrayList<SimpleMatrix> validationExpected = inputs.getValidationExpected();
		ArrayList<SimpleMatrix> allLayers = neuralNetwork.getAllLayers(); 
		
		double[] validationTotalMax = inputs.getValidationTotalMax();
		double[] validationTotalMin = inputs.getValidationTotalMin();
		double[] validationPredictandMax = inputs.getValidationPredictandMax();
		double[] validationPredictandMin = inputs.getValidationPredictandMin();
		
		validationData = standardise(validationData, validationTotalMax, validationTotalMin);


		//this loop works out the sum of the squares of all (Expected Values - Current Values)/Current Values
		double MSREsum = 0; //Value of MSRE before it gets divided by 1/n
		for (int i = 0; i <= inputs.getValidationSize(); i++) {
			
			allLayers.set(0, validationData.get(i));			
			allLayers = forwardPass(allLayers, neuralNetwork);
			
			deStandardise(allLayers.get(networkSize), validationPredictandMax[0], validationPredictandMin[0]);
			MSREsum = MSREsum + Math.pow(((validationExpected.get(i).get(0,0) - allLayers.get(networkSize).get(0, 0))/allLayers.get(networkSize).get(0, 0)), 2);
			standardise(allLayers.get(networkSize), validationPredictandMax[0], validationPredictandMin[0]);
			

		}
		validationData = deStandardise(validationData, validationTotalMax, validationTotalMin);
		double MSRE =  (1/(double) inputs.getValidationSize())*MSREsum;			
		return(MSRE);
	}

	//Runs testing set, and prints graphs based on results
	private static void runTest(inputData inputs, NeuralNetwork neuralNetwork, int networkSize) {
		
		ArrayList<SimpleMatrix> testingExpected = inputs.getTestingExpected();
		ArrayList<SimpleMatrix> testingData = inputs.getTesting();
		
		double[] maxValueTotal = inputs.getTrainingTotalMax(); //We use training maxes and mins, as we arent meant to use testing ones
		double[] minValueTotal = inputs.getTrainingTotalMin();
		double[] maxValuePredictand = inputs.getTrainingPredictandMax();
		double[] minValuePredictand = inputs.getTrainingPredictandMin();
		
		testingData = standardise(testingData, maxValueTotal, minValueTotal);
		ArrayList<SimpleMatrix> allLayers = neuralNetwork.getAllLayers(); 
						
		XYSeries CAESeries = new XYSeries("");  
		
		for (int i = 0; i <= inputs.getTestingSize(); i++) {
			allLayers.set(0, testingData.get(i));
			allLayers = forwardPass(allLayers, neuralNetwork);
			
			deStandardise(allLayers.get(networkSize), maxValuePredictand[0], minValuePredictand[0]);
			CAESeries.add(testingExpected.get(i).get(0,0), allLayers.get(networkSize).get(0, 0)); //Destandardised so it works on the graph
			standardise(allLayers.get(networkSize), maxValuePredictand[0], minValuePredictand[0]);

		}
		
		XYSeriesCollection currentAgainstExpected = new XYSeriesCollection();
		currentAgainstExpected.addSeries(CAESeries);
		
		currentAgainstExpectedChart absoluteChart = new currentAgainstExpectedChart("Expected", currentAgainstExpected);
		absoluteChart.pack();
		RefineryUtilities.centerFrameOnScreen( absoluteChart );

	    absoluteChart.setVisible( true );	
		
	}
	
	private static ArrayList<SimpleMatrix> forwardPass(ArrayList<SimpleMatrix> layers, NeuralNetwork neuralNetwork) {		
		
		// Start from 1 because we don't perform operations on the input layer.
		for (int i = 1; i < layers.size(); i++) {
			layers.set(i, sigmoids(
					layers.get(i - 1).mult(neuralNetwork.getWeights(i - 1)).plus(neuralNetwork.getBiases(i - 1)))); 
		}
		return layers;
	}

	// Standardises all data to the range [0.1, 0.9]
	public static ArrayList<SimpleMatrix> standardise(ArrayList<SimpleMatrix> input, double[] max, double[] min) {
		for (int i = 0; i < input.size(); i++) { // Loops through all rows in ArrayList
			for (int j = 0; j < input.get(i).numCols(); j++) { // Loops through all columns in 1D matrix
				SimpleMatrix currentRow = input.get(i);
				DMatrixIterator it = currentRow.iterator(false, 0, j, currentRow.numRows() - 1, j); // Iterate through
																									// current column
				while (it.hasNext()) {
					it.set((0.8 * (it.next() - min[j]) / (max[j] - min[j])) + 0.1);
				}
				input.set(i, currentRow);
			}
		}
		return input;
	}
	
	//Second standardise function with polymorphism to allow for us standardising single matrices at a time
	public static SimpleMatrix standardise(SimpleMatrix input, double max, double min) {
		DMatrixIterator it = input.iterator(false, 0, 0, input.numRows() - 1, input.numCols() - 1);

		while (it.hasNext()) {
			it.set((0.8 * (it.next() - min) / (max - min)) + 0.1);
		}
		return input;
	}

	// De-standardises the matrix for the correct outputs
	public static ArrayList<SimpleMatrix> deStandardise(ArrayList<SimpleMatrix> input, double[] max, double[] min) {

		for (int i = 0; i < input.size(); i++) { // Loops through all rows in ArrayList
			for (int j = 0; j < input.get(i).numCols(); j++) { // Loops through all columns in 1D matrix
				SimpleMatrix currentRow = input.get(i);
				DMatrixIterator it = currentRow.iterator(false, 0, j, currentRow.numRows() - 1, j); // Iterate through
																									// current column
				while (it.hasNext()) {
					it.set(((it.next() - 0.1) / 0.8) * (max[j] - min[j]) + min[j]);
				}
				input.set(i, currentRow);
			}
		}
		return input;
	}

	//Like standardise, an alternate function to standardise 1 SimpleMatrix at a time
	public static SimpleMatrix deStandardise(SimpleMatrix input, double max, double min) {
		DMatrixIterator it = input.iterator(false, 0, 0, input.numRows() - 1, input.numCols() - 1);

		while (it.hasNext()) {
			it.set(((it.next() - 0.1) / 0.8) * (max - min) + min);
		}
		return input;
	}

	
	
	//Updates the values in the neural network
	public static NeuralNetwork updateValues(int currentEpoch, int currentDataRow, int networkSize, NeuralNetwork neuralNetwork,
			ArrayList<SimpleMatrix> allLayers, double stepSize, HashMap<Integer, SimpleMatrix> deltaMatrices) {
		
		ArrayList<SimpleMatrix> currentWeightChanges = new ArrayList<SimpleMatrix>();
		ArrayList<SimpleMatrix> currentBiasChanges = new ArrayList<SimpleMatrix>();

		
		double momentumMultiplier = 0.8;
		for (int i = 0; i < networkSize; i++) {
			
			SimpleMatrix currentWeightChange = deltaMatrices.get(i + 1).transpose().mult(allLayers.get(i)).scale(stepSize).transpose();
			
			SimpleMatrix currentBiasChange = deltaMatrices.get(i + 1).scale(stepSize);

			if(i == 0 && currentEpoch == 0) {
				

			}
			if(currentEpoch > 0) { /* This is commented out, but if you remove this comment you will be able to use the momentum calculations in the program
				SimpleMatrix lastWeightChanges = neuralNetwork.getLastWeightChanges().get(currentDataRow).get(i).scale(momentumMultiplier);				
				currentWeightChange = currentWeightChange.plus(lastWeightChanges);
				
				SimpleMatrix lastBiasChanges = neuralNetwork.getLastBiasChanges().get(currentDataRow).get(i).scale(momentumMultiplier);				
				currentBiasChange = currentBiasChange.plus(lastBiasChanges);*/
			}
			//System.out.println(i);						
			currentWeightChanges.add(currentWeightChange);
			currentBiasChanges.add(currentBiasChange);
			
			
			SimpleMatrix weightChange = neuralNetwork.getWeights(i)
					.plus(currentWeightChange);
			
			SimpleMatrix biasChange = neuralNetwork.getBiases(i)
					.plus(currentBiasChange);
			
			neuralNetwork.setWeights(i, weightChange);
			neuralNetwork.setBiases(i, biasChange);
		}
		
		neuralNetwork.addCurrentWeightChanges(currentWeightChanges);
		neuralNetwork.addCurrentBiasChanges(currentBiasChanges);
		//neuralNetwork.getCurrentWeightChanges().get(0).get(0).print();
		return neuralNetwork;
	}

	//Updates bias for backprop
	public static SimpleMatrix updateBias(SimpleMatrix bias, double stepSize, SimpleMatrix delta) {
		return bias.plus(delta.scale(stepSize));
	}

	//Updates weights for backprop
	public static SimpleMatrix updateWeight(SimpleMatrix weight, double stepSize, SimpleMatrix delta,
			SimpleMatrix input) {
		
		return weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());

	}

	//Applies sigmoid across entire matrix
	private static SimpleMatrix sigmoids(SimpleMatrix m) {
		DMatrixIterator it = m.iterator(false, 0, 0, m.numRows() - 1, m.numCols() - 1);

		// Iterates through the matrix and applies sigmoid
		while (it.hasNext()) {
			it.set(1 / (1 + Math.exp(-it.next())));
		}

		return m;
	}
	
	//Applies tanh across entire matrix
	private static SimpleMatrix tan(SimpleMatrix m) {
		DMatrixIterator it = m.iterator(false, 0, 0, m.numRows() - 1, m.numCols() - 1);

			// Iterates through the matrix and applies tanh
		while (it.hasNext()) {
			double x = it.next();
			it.set((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)));
		}

		return m;
	}
		

	//Calculates the delta values
	public static SimpleMatrix deltaVal(SimpleMatrix currentDifferentialMatrix, SimpleMatrix weights,
			SimpleMatrix lastDeltaMatrix) {
		
		
		SimpleMatrix averageDeltaWeight = new SimpleMatrix(weights.numRows(), 1); //lastDeltaMatrix.numCols()
		
		//weights.extractVector(false, 0).mult(lastDeltaMatrix.extractVector(false, 0)).print();
		for (int i = 0; i < weights.numCols(); i++) {
			
			averageDeltaWeight = averageDeltaWeight.plus((weights.extractVector(false, i).mult(lastDeltaMatrix.extractVector(false, i))));
			
		}
		averageDeltaWeight = averageDeltaWeight.scale(1 / (double) weights.numCols());
		
		return currentDifferentialMatrix.elementMult(averageDeltaWeight.transpose());
	
	}

	private static SimpleMatrix firstDifferential(SimpleMatrix currentLayer) {
		DMatrixIterator it = currentLayer.iterator(false, 0, 0, currentLayer.numRows() - 1, currentLayer.numCols() - 1);
		// Iterates through the matrix and applies sigmoid/tanh
		
		while (it.hasNext()) {
			double curVal = it.next();
			it.set(curVal * (1 - curVal));   //1 - Math.pow(curVal, 2) <- for sigmoid  
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

			int width = datatypeSheet.getRow(0).getLastCellNum();

			int datasetDeterminant = 1;

			double[] totalRowMaxValidation = new double[width - 1];
			double[] totalRowMinValidation = new double[width - 1];
			double[] predictandMaxValidation = new double[1];
			double[] predictandMinValidation = new double[1];
			predictandMinValidation[0] = Double.MAX_VALUE;

			double[] totalRowMaxTraining = new double[width - 1];
			double[] totalRowMinTraining = new double[width - 1];
			double[] predictandMaxTraining = new double[1];
			double[] predictandMinTraining = new double[1];
			predictandMinTraining[0] = Double.MAX_VALUE;

			for (int i = 0; i < totalRowMinValidation.length; i++) { // Makes all variables in the predictand min column
																		// the max value, so we dont accidentally put a
																		// value lower than minimum in here
				totalRowMinValidation[i] = Double.MAX_VALUE;
			}
			for (int i = 0; i < totalRowMinTraining.length; i++) { // Makes all variables in the predictand min column
																	// the max value, so we dont accidentally put a
																	// value lower than minimum in here
				totalRowMinTraining[i] = Double.MAX_VALUE;
			}

			Iterator<Row> widthIterator = datatypeSheet.iterator();

			int trainingSize = 0;
			int validationSize = 0;
			int testingSize = 0;
			while (widthIterator.hasNext()) {

				Row currentRow = widthIterator.next();
				Iterator<Cell> cellIterator = currentRow.iterator();

				// CHANGE THIS FROM ARRAYLIST (so instead of new double[1][width-1] we set
				// double[i][width-1]
				double[][] currentRowArray = new double[1][width - 1];
				double[][] currentRowPredictedArray = new double[1][1]; // 1x1 array for the output

				int i = 0;

				while (cellIterator.hasNext()) {

					Cell currentCell = cellIterator.next();

					if (i == width - 1) { //if final column (expected output)
						double currentVal = currentCell.getNumericCellValue();
						currentRowPredictedArray[0][0] = currentVal;

						if (datasetDeterminant <= 3) { // First 3 variables go training
							//Compares last value to current value to check for max & minimums
							predictandMaxTraining[0] = Math.max(predictandMaxTraining[0], currentVal); 																														
							predictandMinTraining[0] = Math.min(predictandMinTraining[0], currentVal); 
							
						} else if (datasetDeterminant <= 4) { // First 3 variables go training
							
							predictandMaxValidation[0] = Math.max(predictandMaxValidation[0], currentVal); 
							predictandMinValidation[0] = Math.min(predictandMinValidation[0], currentVal);																				
						}

					} else {
						double currentVal = currentCell.getNumericCellValue();
						currentRowArray[0][i] = currentVal;

						if (datasetDeterminant <= 3) { // First 3 variables go training
							totalRowMaxTraining[i] = Math.max(totalRowMaxTraining[i], currentVal); 																									
							totalRowMinTraining[i] = Math.min(totalRowMinTraining[i], currentVal); 
						} else if (datasetDeterminant <= 4) {
							totalRowMaxValidation[i] = Math.max(totalRowMaxValidation[i], currentVal); 
							totalRowMinValidation[i] = Math.min(totalRowMinValidation[i], currentVal); 
						}

					}

					i++;

				}

				allInputs.addMaxMins(totalRowMaxTraining, totalRowMinTraining, predictandMaxTraining,
						predictandMinTraining, totalRowMaxValidation, totalRowMinValidation, predictandMaxValidation,
						predictandMinValidation);

				SimpleMatrix currentRowMatrix = new SimpleMatrix(currentRowArray);
				SimpleMatrix currentRowPredictedMatrix = new SimpleMatrix(currentRowPredictedArray);

				if (datasetDeterminant <= 3) { // First 3 variables go training
					allInputs.addTraining(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
					trainingSize++;
				} else if (datasetDeterminant <= 4) { // Next row for validation
					allInputs.addValidation(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant++;
					validationSize++;
				} else { // Final row for testing data, then we reset the index to repeat the spread
					allInputs.addTesting(currentRowMatrix, currentRowPredictedMatrix);
					datasetDeterminant = 1;
					testingSize++;
				}

			}
			allInputs.setTrainingSize(trainingSize-1);
			allInputs.setValidationSize(validationSize-1);
			allInputs.setTestingSize(testingSize-1);
			workbook.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return allInputs;

	}

}
