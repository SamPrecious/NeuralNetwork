package uk.ac.lboro.COB107.NeuralNetwork;

import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.plot.PlotOrientation;

public class ChartPlotter extends ApplicationFrame {

	
	private static final long serialVersionUID = 1L;

	public ChartPlotter(String title, String chartTitle, XYDataset dataSet, String xAxis, String yAxis) {
		super(title);

		JFreeChart lineChart = ChartFactory.createXYLineChart(chartTitle, xAxis, yAxis, dataSet, PlotOrientation.VERTICAL, false, false, false);
		
		//lineChart.getXYPlot().setRenderer(new XYSplineRenderer());
		ChartPanel chartPanel = new ChartPanel(lineChart);
		chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
		setContentPane(chartPanel);
		

	}

}
