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

		
		/*
		private void createInputData() {
			training = new ArrayList<SimpleMatrix>();
			validation = new ArrayList<SimpleMatrix>();
			testing = new ArrayList<SimpleMatrix>();

		}*/
		
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
		
}	
