package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class inputData {
		
		private ArrayList<SimpleMatrix> training = new ArrayList<SimpleMatrix>();
		private ArrayList<SimpleMatrix> trainingExpected = new ArrayList<SimpleMatrix>();
		
		private ArrayList<SimpleMatrix> validation = new ArrayList<SimpleMatrix>();
		private ArrayList<SimpleMatrix> validationExpected = new ArrayList<SimpleMatrix>();

		
		private ArrayList<SimpleMatrix> testing = new ArrayList<SimpleMatrix>();
		private ArrayList<SimpleMatrix> testingExpected = new ArrayList<SimpleMatrix>();
		
		private int trainingSize;
		
		private double[] trainingTotalMax;
		private double[] trainingTotalMin;
		private double[] trainingPredictandMax;
		private double[] trainingPredictandMin;
		
		
		private double[] validationTotalMax;
		private double[] validationTotalMin;
		private double[] validationPredictandMax;
		private double[] validationPredictandMin;
						
		
		
		
		public void addTrainingSize(int trainingSize) {
			this.trainingSize = trainingSize;
		}
		public int getTrainingSize() {
			return trainingSize;
		}
		
		public void addTraining(SimpleMatrix newTrainingData, SimpleMatrix expected) {
			training.add(newTrainingData);
			trainingExpected.add(expected);
		}
		public void addValidation(SimpleMatrix newValidationData, SimpleMatrix expected) {
			validation.add(newValidationData);
			validationExpected.add(expected);
		}
		public void addTesting(SimpleMatrix newTestingData, SimpleMatrix expected) {
			testing.add(newTestingData);
			testingExpected.add(expected);
		}
		
		public ArrayList<SimpleMatrix> getTraining() {
			return training;
		}
		
		public ArrayList<SimpleMatrix> getTrainingExpected() {
			return trainingExpected;
		}
		
		public ArrayList<SimpleMatrix> getValidation() {
			return validation;
		}
		public ArrayList<SimpleMatrix> getValidationExpected() {
			return validationExpected;
		}
		
		public ArrayList<SimpleMatrix> getTesting() {
			return testing;
		}
		public ArrayList<SimpleMatrix> getTestingExpected() {
			return testingExpected;
		}
		
		
		public void addMaxMins(double[] trainingTotalMax, double[] trainingTotalMin, double[] trainingPredictandMax, double[] trainingPredictandMin, double[] validationTotalMax, double[] validationTotalMin, double[] validationPredictandMax, double[] validationPredictandMin) {
			this.trainingTotalMax = trainingTotalMax;
			this.trainingTotalMin = trainingTotalMin;
			this.trainingPredictandMax = trainingPredictandMax;
			this.trainingPredictandMin  = trainingPredictandMin;
			
			this.validationTotalMax = validationTotalMax;
			this.validationTotalMin = validationTotalMin;
			this.validationPredictandMax = validationPredictandMax;
			this.validationPredictandMin = validationPredictandMin;
		}
		
		public double[] getTrainingTotalMax() {
			return trainingTotalMax;
		}
		public double[] getTrainingTotalMin() {
			return trainingTotalMin;
		}
		public double[] getTrainingPredictandMax() {
			return trainingPredictandMax;
		}
		public double[] getTrainingPredictandMin() {
			return trainingPredictandMin;
		}
		public double[] getValidationTotalMax() {
			return validationTotalMax;
		}
		public double[] getValidationTotalMin() {
			return validationTotalMin;
		}
		public double[] getValidationPredictandMax() {
			return validationPredictandMax;
		}
		public double[] getValidationPredictandMin() {
			return validationPredictandMin;
		}
		
		
		
		
		
		
		
}	
