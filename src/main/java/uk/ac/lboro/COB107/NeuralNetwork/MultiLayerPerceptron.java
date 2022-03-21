package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;

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

	public static void main(String[] args) {

		MultiLayerPerceptron predict = new MultiLayerPerceptron();

		inputData allInputs = getInputs();

		predict(allInputs);
	}

	

	public static void predict(inputData allInputs) {
		
		
		
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
		SimpleMatrix weightMatrix1 = new SimpleMatrix(weightArray1);

		double[][] biasArray1 = { { 1, -6 } };
		SimpleMatrix biasMatrix1 = new SimpleMatrix(biasArray1);

		double[][] weightArray2 = { { 2 }, { 4 } };
		SimpleMatrix weightMatrix2 = new SimpleMatrix(weightArray2);

		double[][] biasArray2 = { { -3.92 } };
		SimpleMatrix biasMatrix2 = new SimpleMatrix(biasArray2);

		SimpleMatrix hiddenNode = new SimpleMatrix(inputMatrix.numRows(), weightMatrix1.numCols());
		SimpleMatrix outputNode = new SimpleMatrix(hiddenNode.numRows(), weightMatrix2.numCols());

		double outputDelta;
		double[] hiddenDelta = new double[outputNode.numCols()];

		double stepSize = 0.1;

		double correctOutput = 1;

		// Initially do a forward pass to start off with
		hiddenNode = forwardPass(hiddenNode, inputMatrix, weightMatrix1, biasMatrix1);
		outputNode = forwardPass(outputNode, hiddenNode, weightMatrix2, biasMatrix2);

		// outputNode.print();

		// This is our main loop, doing a forward pass, then a backward pass, for x
		// number of epochs
		for (int i = 0; i < 20000; i++) {

			// The rest is a backward pass, we need most values in this function, so there
			// is no point creating another function for this.

			outputDelta = (correctOutput - outputNode.get(0, 0)) * (outputNode.get(0, 0) * (1 - outputNode.get(0, 0)));

			double[][] outputDeltaArray = { { outputDelta } };
			SimpleMatrix outputDeltaMatrix = new SimpleMatrix(outputDeltaArray);

			// Here we clone our hidden node matrix so we can get a matrix of current
			// differentials
			SimpleMatrix hiddenDeltaMatrix = hiddenNode.copy();
			firstDifferential(hiddenDeltaMatrix);
			hiddenDeltaMatrix = deltaVal(hiddenDeltaMatrix, weightMatrix2, outputDeltaMatrix);

			// New weight: Old Weight + (Step Size * currentDelta * input)

			weightMatrix1 = updateWeight(weightMatrix1, stepSize, hiddenDeltaMatrix, inputMatrix);
			biasMatrix1 = updateBias(biasMatrix1, stepSize, hiddenDeltaMatrix);
			weightMatrix2 = updateWeight(weightMatrix2, stepSize, outputDeltaMatrix, hiddenNode);
			biasMatrix2 = updateBias(biasMatrix2, stepSize, outputDeltaMatrix);

			// Apply forward pass with updated values
			hiddenNode = forwardPass(hiddenNode, inputMatrix, weightMatrix1, biasMatrix1);
			outputNode = forwardPass(outputNode, hiddenNode, weightMatrix2, biasMatrix2);

			// outputNode.print();

		}

	}

	public static SimpleMatrix updateBias(SimpleMatrix bias, double stepSize, SimpleMatrix delta) {
		// We want to get oldVal = oldVal + (stepSize*delta)
		return bias.plus(delta.scale(stepSize));
	}

	public static SimpleMatrix updateWeight(SimpleMatrix weight, double stepSize, SimpleMatrix delta,
			SimpleMatrix input) {
		/*
		 * We want to get oldVal = oldVal + (stepSize*delta*input) This matrix
		 * calculation works out these values
		 */

		return weight.plus(delta.transpose().mult(input).scale(stepSize).transpose());

	}

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

	// Performs the forward through layer in a network
	private static SimpleMatrix forwardPass(SimpleMatrix currentLayer, SimpleMatrix input, SimpleMatrix weight,
			SimpleMatrix bias) {

		currentLayer = input.mult(weight).plus(bias);

		sigmoid(currentLayer);

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
