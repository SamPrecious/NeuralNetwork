package uk.ac.lboro.COB107.NeuralNetwork;

import java.io.Serializable; 
import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork implements Serializable{ //Serializable allows us to write the Neural Network to a file

	// multiple layers. Each layer has x weights relating to nodes

	private ArrayList<SimpleMatrix> allLayers = new ArrayList<SimpleMatrix>();

	private ArrayList<SimpleMatrix> allBiases = new ArrayList<SimpleMatrix>();
	private ArrayList<SimpleMatrix> allWeights = new ArrayList<SimpleMatrix>();
	private ArrayList<ArrayList<SimpleMatrix>> lastWeightChanges = new ArrayList<ArrayList<SimpleMatrix>>();
	private ArrayList<ArrayList<SimpleMatrix>> currentWeightChanges = new ArrayList<ArrayList<SimpleMatrix>>();

	private ArrayList<ArrayList<SimpleMatrix>> lastBiasChanges = new ArrayList<ArrayList<SimpleMatrix>>();
	private ArrayList<ArrayList<SimpleMatrix>> currentBiasChanges = new ArrayList<ArrayList<SimpleMatrix>>();

	
	public void addBiases(SimpleMatrix bias) {
		allBiases.add(bias);
	}

	public SimpleMatrix getBiases(int index) {
		return allBiases.get(index);
	}

	public ArrayList<SimpleMatrix> getAllLayers() {
		return allLayers;
	}

	public void setAllLayers(ArrayList<SimpleMatrix> allLayers) {
		this.allLayers = allLayers;
	}

	public void setLastWeightChanges(ArrayList<ArrayList<SimpleMatrix>> weightChanges) {
		this.lastWeightChanges = weightChanges;
	}
	
	
	public ArrayList<ArrayList<SimpleMatrix>> getLastWeightChanges(){
		return lastWeightChanges;
	}
	
	public void setLastBiasChanges(ArrayList<ArrayList<SimpleMatrix>> lastBiasChanges) {
		this.lastBiasChanges = lastBiasChanges;
	}
	
	
	public ArrayList<ArrayList<SimpleMatrix>> getLastBiasChanges(){
		return lastBiasChanges;
	}
	
	
	public void clearCurrentWeightChanges() {
		currentWeightChanges.clear();
	}
	
	public void clearCurrentBiasChanges() {
		currentBiasChanges.clear();
	}
	
	public void setCurrentWeightChanges(ArrayList<ArrayList<SimpleMatrix>> weightChanges) {
		this.currentWeightChanges = weightChanges;
	}
	
	public void setCurrentBiasChanges(ArrayList<ArrayList<SimpleMatrix>> biasChanges) {
		this.currentBiasChanges = biasChanges;
	}
	
	public void addCurrentWeightChanges(ArrayList<SimpleMatrix> weightChanges) {
		this.currentWeightChanges.add(weightChanges);
	}
	
	public void addCurrentBiasChanges(ArrayList<SimpleMatrix> biasChanges) {
		this.currentBiasChanges.add(biasChanges);
	}
	
	public ArrayList<ArrayList<SimpleMatrix>> getCurrentWeightChanges(){
		return currentWeightChanges;
	}
	public ArrayList<ArrayList<SimpleMatrix>> getCurrentBiasChanges(){
		return currentBiasChanges;
	}
	
	public void setBiases(int index, SimpleMatrix bias) {
		allBiases.set(index, bias);
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
