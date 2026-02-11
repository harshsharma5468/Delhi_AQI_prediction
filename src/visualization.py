import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AQIVisualizer:
    def __init__(self, df=None):
        self.df = df
        self.set_style()
    
    def set_style(self):
        """Set visualization style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Plotly template
        self.template = 'plotly_white'
    
    def time_series_plot(self, pollutant='pm2_5', start_date=None, end_date=None):
        """
        Create time series plot for a pollutant
        
        Parameters:
        pollutant: name of pollutant to plot
        start_date: start date for filtering
        end_date: end date for filtering
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        df_plot = self.df.copy()
        
        if start_date:
            df_plot = df_plot[df_plot['date'] >= start_date]
        if end_date:
            df_plot = df_plot[df_plot['date'] <= end_date]
        
        fig = go.Figure()
        
        # Add main trace
        fig.add_trace(go.Scatter(
            x=df_plot['date'],
            y=df_plot[pollutant],
            mode='lines',
            name=pollutant,
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        # Add rolling average
        window = 24  # 24-hour rolling average
        df_plot['rolling_avg'] = df_plot[pollutant].rolling(window=window, center=True).mean()
        
        fig.add_trace(go.Scatter(
            x=df_plot['date'],
            y=df_plot['rolling_avg'],
            mode='lines',
            name=f'{window}-hour average',
            line=dict(color='#1E3A8A', width=3, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Time Series of {pollutant.upper()}',
            xaxis_title='Date',
            yaxis_title=f'{pollutant.upper()} Concentration (μg/m³)',
            hovermode='x unified',
            template=self.template,
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def pollutant_comparison_plot(self, pollutants=None):
        """
        Compare multiple pollutants
        
        Parameters:
        pollutants: list of pollutants to compare
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        if pollutants is None:
            pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
        
        # Calculate statistics
        stats = pd.DataFrame({
            'Mean': self.df[pollutants].mean(),
            'Median': self.df[pollutants].median(),
            'Std': self.df[pollutants].std(),
            'Max': self.df[pollutants].max(),
            'Min': self.df[pollutants].min()
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Concentration', 'Maximum Values', 
                          'Standard Deviation', 'Box Plot'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'box'}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        # Mean concentration
        fig.add_trace(
            go.Bar(x=stats.index, y=stats['Mean'], name='Mean', marker_color=colors[0]),
            row=1, col=1
        )
        
        # Maximum values
        fig.add_trace(
            go.Bar(x=stats.index, y=stats['Max'], name='Max', marker_color=colors[1]),
            row=1, col=2
        )
        
        # Standard deviation
        fig.add_trace(
            go.Bar(x=stats.index, y=stats['Std'], name='Std Dev', marker_color=colors[2]),
            row=2, col=1
        )
        
        # Box plot
        for i, pollutant in enumerate(pollutants):
            fig.add_trace(
                go.Box(y=self.df[pollutant], name=pollutant, marker_color=colors[i % len(colors)]),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Pollutant Comparison',
            template=self.template,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def correlation_heatmap(self, pollutants=None):
        """
        Create correlation heatmap for pollutants
        
        Parameters:
        pollutants: list of pollutants to include
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        if pollutants is None:
            pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3']
        
        # Calculate correlation matrix
        corr_matrix = self.df[pollutants].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Pollutant Correlation Matrix',
            template=self.template,
            height=600,
            xaxis_title='Pollutants',
            yaxis_title='Pollutants'
        )
        
        return fig
    
    def hourly_pattern_plot(self, pollutant='pm2_5'):
        """
        Show hourly pattern of a pollutant
        
        Parameters:
        pollutant: pollutant to analyze
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        # Group by hour
        hourly_stats = self.df.groupby('hour')[pollutant].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color='#1E3A8A', width=3),
            marker=dict(size=8)
        ))
        
        # Add confidence interval (mean ± std)
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['mean'] + hourly_stats['std'],
            mode='lines',
            name='Mean + Std',
            line=dict(color='rgba(30, 58, 138, 0.2)', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['mean'] - hourly_stats['std'],
            mode='lines',
            name='Mean - Std',
            line=dict(color='rgba(30, 58, 138, 0.2)', width=1),
            fill='tonexty',
            fillcolor='rgba(30, 58, 138, 0.1)',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Hourly Pattern of {pollutant.upper()}',
            xaxis_title='Hour of Day',
            yaxis_title=f'{pollutant.upper()} Concentration (μg/m³)',
            template=self.template,
            height=500,
            xaxis=dict(tickmode='linear', dtick=2)
        )
        
        return fig
    
    def aqi_distribution_plot(self):
        """
        Show distribution of PM2.5 with AQI categories
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        # Define AQI categories
        bins = [0, 30, 60, 90, 120, 250, float('inf')]
        labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
        
        # Categorize PM2.5 values
        self.df['aqi_category'] = pd.cut(
            self.df['pm2_5'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Count by category
        category_counts = self.df['aqi_category'].value_counts().reindex(labels)
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='inside'
        )])
        
        # Update layout
        fig.update_layout(
            title='Distribution of AQI Categories',
            template=self.template,
            height=500,
            annotations=[dict(
                text=f'Total Samples<br>{len(self.df)}',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )]
        )
        
        return fig
    
    def scatter_matrix_plot(self, pollutants=None):
        """
        Create scatter matrix for pollutants
        
        Parameters:
        pollutants: list of pollutants to include
        """
        if self.df is None:
            print("Data not loaded.")
            return None
        
        if pollutants is None:
            pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            self.df,
            dimensions=pollutants,
            title='Scatter Matrix of Pollutants',
            height=800
        )
        
        # Update diagonal
        fig.update_traces(diagonal_visible=False)
        
        return fig
    
    def forecast_plot(self, actual, predicted, dates):
        """
        Create forecast vs actual plot
        
        Parameters:
        actual: actual values
        predicted: predicted values
        dates: corresponding dates
        """
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='#1E3A8A', width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        # Add confidence interval (for demonstration)
        upper_bound = predicted * 1.1
        lower_bound = predicted * 0.9
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_bound,
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.1)',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Actual vs Predicted PM2.5',
            xaxis_title='Date',
            yaxis_title='PM2.5 Concentration (μg/m³)',
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('data/delhi_aqi.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize visualizer
    visualizer = AQIVisualizer(df)
    
    # Create visualizations
    fig1 = visualizer.time_series_plot('pm2_5')
    fig2 = visualizer.pollutant_comparison_plot()
    fig3 = visualizer.correlation_heatmap()
    fig4 = visualizer.hourly_pattern_plot('pm2_5')
    fig5 = visualizer.aqi_distribution_plot()
    
    # Save figures
    fig1.write_html('visualizations/time_series.html')
    fig2.write_html('visualizations/pollutant_comparison.html')
    fig3.write_html('visualizations/correlation_heatmap.html')
    
    print("Visualizations created successfully!")