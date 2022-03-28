package uk.ac.lboro.COB107.NeuralNetwork;

import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYSplineRenderer;

public class ChartPlotter extends ApplicationFrame {

	public ChartPlotter(String title, String chartTitle, XYDataset dataSet) {
		super(title);

		JFreeChart lineChart = ChartFactory.createXYLineChart(chartTitle, "Epochs", "Absolute Error", dataSet, PlotOrientation.VERTICAL, false, false, false);
		
		//lineChart.getXYPlot().setRenderer(new XYSplineRenderer());
		ChartPanel chartPanel = new ChartPanel(lineChart);
		chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
		setContentPane(chartPanel);
		
		
		/*
		JFreeChart lineChart = ChartFactory.createLineChart(chartTitle, "Epochs", "Absolute Error", dataSet,
				PlotOrientation.VERTICAL, true, true, false);
		
		
		lineChart.getXYPlot().setRenderer(new XYSplineRenderer());
		
		ChartPanel chartPanel = new ChartPanel(lineChart);
		chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
		setContentPane(chartPanel);*/

	}

}
