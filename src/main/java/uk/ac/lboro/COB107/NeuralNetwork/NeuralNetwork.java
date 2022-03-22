package uk.ac.lboro.COB107.NeuralNetwork;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {


	
	private ArrayList<SimpleMatrix> nodes = new ArrayList<SimpleMatrix>();

	
	private ArrayList<SimpleMatrix> allBiases = new ArrayList<SimpleMatrix>();
	private ArrayList<SimpleMatrix> allWeights = new ArrayList<SimpleMatrix>();

	
	
	
	
	public void addNode(SimpleMatrix node) {
		nodes.add(node);
	}
	
	public SimpleMatrix getNode(int index) {
		return nodes.get(index);
	}
	
	public void setNode(int index, SimpleMatrix node) {
		nodes.set(index, node);
	}
	
	
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
	
	public SimpleMatrix getWeights(int index) {
		return allWeights.get(index);
	}
	
	public int getWeightLength() {
		return allWeights.size();
	}
}
