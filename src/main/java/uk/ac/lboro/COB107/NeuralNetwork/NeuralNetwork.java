package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {

	//multiple layers. Each layer has x weights relating to nodes
	

	
	private ArrayList<SimpleMatrix> allBiases = new ArrayList<SimpleMatrix>();
	private ArrayList<SimpleMatrix> allWeights = new ArrayList<SimpleMatrix>();
	
	
	public void addBiases(SimpleMatrix bias) {
		allBiases.add(bias);
	}
	
	public SimpleMatrix getBiases(int index) {
		return allBiases.get(index);
	}
	
	public int getBiasLength() {
		return allBiases.size();
	}
	
	
	
	public void addWeights(SimpleMatrix weight) {
		allWeights.add(weight);
	}
	
	public void setWeights(int index, SimpleMatrix weight) {
		allWeights.set(index, weight);
	}
	
	public SimpleMatrix getWeights(int index) {
		return allWeights.get(index);
	}
	
	public int getWeightLength() {
		return allWeights.size();
	}
}
