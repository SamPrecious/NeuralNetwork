package uk.ac.lboro.COB107.NeuralNetwork;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.ApplicationFrame;

public class currentAgainstExpectedChart extends ApplicationFrame {

	public currentAgainstExpectedChart(String title, String chartTitle, XYDataset dataSet) {
		super(title);

		JFreeChart lineChart = ChartFactory.createScatterPlot(chartTitle, "Expected", "Given", dataSet,
				PlotOrientation.VERTICAL, false, false, false);

		// lineChart.getXYPlot().setRenderer(new XYSplineRenderer());
		ChartPanel chartPanel = new ChartPanel(lineChart);
		chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
		setContentPane(chartPanel);
	}

}
