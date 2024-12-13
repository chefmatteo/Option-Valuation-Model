import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_authenticator as stauth
import yfinance as yf
import warnings
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend

# Suppress specific warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext! This warning can be ignored when running in bare mode.")

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.call_price = None
        self.put_price = None

    def calculate_prices(self):
        d1 = (
            log(self.current_price / self.strike) +
            (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (self.volatility * sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        self.call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        )
        self.put_price = (
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        return self.call_price, self.put_price

    def calculate_price_surface(self, spot_range, time_range):
        Z = np.zeros((len(time_range), len(spot_range)))
        
        for i in range(len(time_range)):
            for j in range(len(spot_range)):
                self.current_price = spot_range[j]
                self.time_to_maturity = time_range[i]
                self.calculate_prices()
                Z[i, j] = self.call_price  # Store the calculated call price directly
        
        return Z


    def plot_3d_surface(self, spot_range, time_range):
        Z = self.calculate_price_surface(spot_range, time_range)
        
        # Create the figure and axes
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(spot_range, time_range)

        # Custom hillshading
        ls = LightSource(azdeg=270, altdeg=45)
        rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                                   linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Time to Expiry (Years)')
        ax.set_zlabel('Call Option Price')
        ax.set_title('Black-Scholes Call Option Price Surface with Hillshading')

        # Display the 3D plot in Streamlit
        st.pyplot(fig)

    def plot_heatmaps(self, volatility_range, strike_range):
        call_prices = np.zeros((len(volatility_range), len(strike_range)))
        put_prices = np.zeros((len(volatility_range), len(strike_range)))

        for i, vol in enumerate(volatility_range):
            for j, strike in enumerate(strike_range):
                self.strike = strike
                self.volatility = vol
                self.calculate_prices()
                call_prices[i, j] = self.call_price
                put_prices[i, j] = self.put_price

        # Plotting Call Price Heatmap
        fig_call, ax_call = plt.subplots(figsize=(10, 8))
        sns.heatmap(call_prices, xticklabels=np.round(strike_range, 2), yticklabels=np.round(volatility_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
        ax_call.set_title('Call Option Price Heatmap (Volatility vs Strike Price)')
        ax_call.set_xlabel('Strike Price')
        ax_call.set_ylabel('Volatility')

        # Display the heatmap in Streamlit
        st.pyplot(fig_call)

        # Plotting Put Price Heatmap
        fig_put, ax_put = plt.subplots(figsize=(10, 8))
        sns.heatmap(put_prices, xticklabels=np.round(strike_range, 2), yticklabels=np.round(volatility_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
        ax_put.set_title('Put Option Price Heatmap (Volatility vs Strike Price)')
        ax_put.set_xlabel('Strike Price')
        ax_put.set_ylabel('Volatility')

        # Display the heatmap in Streamlit
        st.pyplot(fig_put)

    def get_prices(self):
        return self.call_price, self.put_price

class MonteCarloOptionPricingModel: 
    def __init__(
        self,
        stock_symbol, 
        time_to_maturity: float, # k 
        strike: float, #
        risk_free_rate: float,
        volatility: float,
        num_simulations: int, 
        num_steps: int
    ):    
        self.stock_symbol = stock_symbol 
        self.time_to_maturity = time_to_maturity
        self.strike = strike 
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.initial_stock_price = self.get_initial_stock_price()
        
        
        
    def get_initial_stock_price(self): 
        stock_data = yf.Ticker(self.stock_symbol)
        return stock_data.history(period = "1d")["Close"].iloc[-1] # obtain the close price of the day from yahoo finance
    
    def simulate_stock_price(self): 
        S0 = self.initial_stock_price
        dt =self.time_to_maturity/ self.num_steps
        prices = np.zeros((self.num_steps + 1, self.num_simulations))
        prices[0] = 50 
        
        for i in range(1, self.num_steps + 1): # coz the final step is being excluded, thus + 1
            z = np.random.normal(size = self.num_simulations)
            prices[i] = prices[i - 1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * z)           

        return prices
    
    def plot_simulations(self, simulated_prices): 
        plt.figure(figsize=(10, 6))
        
        # Plot each simulation path
        for i in range(simulated_prices.shape[1]):  # Iterate over the number of simulations
            plt.plot(simulated_prices[:, i], lw=1, alpha=0.6)
        
        plt.title(f"Monte Carlo Simulation of {self.stock_symbol} Option Prices")
        plt.xlabel("Days")
        plt.ylabel("Price Valuation")
        plt.grid(True)
        plt.xlim(0, self.num_steps)
        plt.ylim(0, np.max(simulated_prices) * 1.1)  # To ensure that the plot can show all the prices
        plt.axhline(y=self.strike, linestyle="--", label="Strike Price")  # Use self.strike instead of self.strike_price
        plt.legend() 
        plt.show()  # Display the plot

class BinomialOptionPricingModel:
    def __init__(self, stock_symbol, time_to_maturity, strike, risk_free_rate, volatility, num_simulations, num_steps, dividend_yield=0, american=False):
        self.stock_symbol = stock_symbol
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.dividend_yield = dividend_yield
        self.american = american
        self.initial_stock_price = self.get_initial_stock_price()

    def get_initial_stock_price(self):
        stock_data = yf.Ticker(self.stock_symbol)
        return stock_data.history(period="1d")["Close"].iloc[-1]

    def simulate_stock_price(self):
        S0 = self.initial_stock_price
        dt = self.time_to_maturity / self.num_steps
        prices = np.zeros((self.num_steps + 1, self.num_simulations))
        prices[0] = S0

        for i in range(1, self.num_steps + 1):
            z = np.random.normal(size=self.num_simulations)
            prices[i] = prices[i - 1] * np.exp((self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * z)

        return prices

    def calculate_option_price(self, simulated_prices):
        option_prices = np.zeros(self.num_simulations)
        for i in range(self.num_simulations):
            if self.american:
                option_prices[i] = self.calculate_american_option_price(simulated_prices[:, i])
            else:
                option_prices[i] = self.calculate_european_option_price(simulated_prices[:, i])

        return np.mean(option_prices)

    def calculate_european_option_price(self, simulated_price_path):
        return np.maximum(simulated_price_path[-1] - self.strike, 0)

    def calculate_american_option_price(self, simulated_price_path):
        option_price = np.zeros(self.num_steps + 1)
        option_price[-1] = np.maximum(simulated_price_path[-1] - self.strike, 0)

        for i in range(self.num_steps - 1, -1, -1):
            option_price[i] = np.maximum(simulated_price_path[i] - self.strike, np.exp(-self.risk_free_rate * self.time_to_maturity / self.num_steps) * option_price[i + 1])

        return option_price[0]

    def plot_simulations(self, simulated_prices):
        plt.figure(figsize=(10, 6))

        for i in range(simulated_prices.shape[1]):
            plt.plot(simulated_prices[:, i], lw=1, alpha=0.6)

        plt.title(f"Binomial Option Pricing Model for {self.stock_symbol}")
        plt.xlabel("Days")
        plt.ylabel("Price Valuation")
        plt.grid(True)
        plt.xlim(0, self.num_steps)
        plt.ylim(0, np.max(simulated_prices) * 1.1)
        plt.axhline(y=self.strike, linestyle="--", label="Strike Price")
        plt.legend()
        plt.show()

    def plot_option_price_distribution(self, option_prices):
        plt.figure(figsize=(10, 6))
        plt.hist(option_prices, bins=50, density=True)
        plt.title(f"Distribution of Option Prices for {self.stock_symbol}")
        plt.xlabel("Option Price")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()
# ... your existing imports and BlackScholes class definition ...

class BNSModel: 
    
    def __init__(self, initial_stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, 
                 jump_intensity, mean_jump_size, jump_size_volatility, correlation):
        """
        Initialize Barndorff-Nielsen and Shephard (BNS) model parameters
        
        Args:
            initial_stock_price (float): Initial stock price
            strike_price (float): Strike price 
            time_to_maturity (float): Time to maturity in years
            risk_free_rate (float): Risk-free interest rate
            volatility (float): Initial volatility
            jump_intensity (float): Intensity of jumps in variance process
            mean_jump_size (float): Mean size of jumps
            jump_size_volatility (float): Volatility of jump sizes
            correlation (float): Correlation between volatility and price processes
        """
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.mean_jump_size = mean_jump_size
        self.jump_size_volatility = jump_size_volatility
        self.correlation = correlation

    def simulate_paths(self, number_of_paths, number_of_steps):
        """
        Simulate stock price paths using BNS model
        
        Args:
            number_of_paths (int): Number of paths to simulate
            number_of_steps (int): Number of time steps
            
        Returns:
            numpy.ndarray: Matrix of simulated prices
        """
        time_step = self.time_to_maturity / number_of_steps
        prices = np.zeros((number_of_steps + 1, number_of_paths))
        prices[0] = self.initial_stock_price
        
        for path in range(number_of_paths):
            # Generate random variables
            wiener_process = np.random.normal(0, np.sqrt(time_step), number_of_steps)
            poisson_process = np.random.poisson(self.jump_intensity * time_step, number_of_steps)
            jump_sizes = np.random.normal(self.mean_jump_size, self.jump_size_volatility, number_of_steps)
            
            # Simulate variance process
            variance = np.zeros(number_of_steps + 1)
            variance[0] = self.volatility**2
            
            for step in range(number_of_steps):
                # Update variance
                variance[step+1] = variance[step] + self.jump_intensity * (self.mean_jump_size - variance[step]) * time_step + \
                                 self.jump_size_volatility * np.sqrt(variance[step]) * wiener_process[step]
                variance[step+1] = max(0, variance[step+1])  # Ensure positive variance
                
                # Update price
                drift_term = (self.risk_free_rate - 0.5 * variance[step]) * time_step
                diffusion_term = np.sqrt(variance[step]) * wiener_process[step]
                jump_term = np.sum(jump_sizes[:step+1] * poisson_process[:step+1])
                
                prices[step+1, path] = prices[step, path] * np.exp(drift_term + diffusion_term + jump_term)
                
        return prices

    def price_european_options(self, number_of_paths=10000, number_of_steps=252):
        """
        Price European call and put options using Monte Carlo simulation
        
        Args:
            number_of_paths (int): Number of simulation paths
            number_of_steps (int): Number of time steps
            
        Returns:
            tuple: (call_option_price, put_option_price)
        """
        simulated_paths = self.simulate_paths(number_of_paths, number_of_steps)
        final_prices = simulated_paths[-1]
        
        # Calculate discounted payoffs
        discount_factor = np.exp(-self.risk_free_rate * self.time_to_maturity)
        call_option_payoffs = np.maximum(final_prices - self.strike_price, 0)
        put_option_payoffs = np.maximum(self.strike_price - final_prices, 0)
        
        # Calculate option prices
        call_option_price = discount_factor * np.mean(call_option_payoffs)
        put_option_price = discount_factor * np.mean(put_option_payoffs)
        
        return call_option_price, put_option_price

    def plot_sample_paths(self, number_of_paths=5, number_of_steps=252):
        """
        Plot sample price paths
        
        Args:
            number_of_paths (int): Number of paths to plot
            number_of_steps (int): Number of time steps
            
        Returns:
            matplotlib.figure.Figure: Plot of sample paths
        """
        simulated_paths = self.simulate_paths(number_of_paths, number_of_steps)
        time_points = np.linspace(0, self.time_to_maturity, number_of_steps + 1)
        
        plt.figure(figsize=(10, 6))
        for path in range(number_of_paths):
            plt.plot(time_points, simulated_paths[:, path])
            
        plt.title('Barndorff-Nielsen and Shephard (BNS) Model Sample Paths')
        plt.xlabel('Time (Years)')
        plt.ylabel('Stock Price')
        plt.grid(True)
        return plt.gcf()


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Options Pricing Model")
    st.write("`Created by:`")
    linkedin_url = "www.linkedin.com/in/matthew-ng-315a07281"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Matthew Ng`</a>', unsafe_allow_html=True)

    st.markdown("### Stock Information")
    stock_symbol = st.text_input("Enter stock symbol (e.g, APPL, TSLA):", "AAPL")
    current_price = st.number_input("Current Asset Price", value=100.0)

    st.markdown("---")
    st.markdown("### Black-Scholes Model Parameters")
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Interest Rate", value=0.05)

    st.markdown("---")
    st.markdown("### Monte Carlo Model Parameters")
    risk_free_rate = st.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.05)
    number_of_simulations = st.number_input("Number of simulations:", min_value=1, value=1000)
    number_of_steps = st.number_input("Number of steps:", min_value=1, value=252)

    st.markdown("---")
    st.markdown("### Binomial Model Parameters")
    dividend_yield = st.number_input("Dividend Yield", min_value=0.0, value=0.0)
    american_option = st.checkbox("American Option", value=False)

    st.markdown("---")
    st.markdown("### Heatmap Parameters")
    calculate_btn = st.button('Generate Heatmap')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

    st.markdown("---")
    st.markdown("### BNS Model Parameters")
    lambda_param = st.number_input("Lambda (Jump Intensity)", min_value=0.0, value=1.0, step=0.1)
    rho = st.number_input("Rho (Mean Reversion)", min_value=0.0, value=0.5, step=0.1)
    nu = st.number_input("Nu (Jump Size)", min_value=0.0, value=0.2, step=0.01)
    theta = st.number_input("Theta (Long-term Variance)", min_value=0.0, value=0.04, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    
    

# Function to generate heatmaps and simulations 

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put





# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike], 
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "3D Plot": ["Enabled"],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Add 3D Surface Plot
st.title("Black-Scholes 3D Surface Plot")
st.info("This 3D surface plot shows how the call option price varies with spot price and time to expiry")

# Create spot and time ranges for 3D plot
spot_range_3d = np.linspace(0.8 * current_price, 1.2 * current_price, 50)
time_range_3d = np.linspace(0.1, time_to_maturity, 50)
X, Y = np.meshgrid(spot_range_3d, time_range_3d)
Z = np.zeros_like(X)

# Calculate option prices for each spot price and time combination
for i in range(len(time_range_3d)):
    for j in range(len(spot_range_3d)):
        temp_bs = BlackScholes(
            time_to_maturity=time_range_3d[i],
            strike=strike,
            current_price=spot_range_3d[j],
            volatility=volatility,
            interest_rate=interest_rate
        )
        Z[i,j], _ = temp_bs.calculate_prices()

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Spot Price')
ax.set_ylabel('Time to Expiry (Years)')
ax.set_zlabel('Call Option Price')
ax.set_title('Black-Scholes Call Option Price Surface')

# Add colorbar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# Display the 3D plot
st.pyplot(fig)

st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    heatmap_fig_put, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    if heatmap_fig_put is not None:
        st.pyplot(heatmap_fig_put)
    else:
        st.error("The heatmap figure could not be created.")
st.title("Barndorff-Nielsen and Shephard (BNS) Model")

# Table of Inputs for BNS Model
input_data_bns = {
    "Stock Symbol": [stock_symbol],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity], 
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [risk_free_rate],
    "Jump Intensity": [0.5],  # Default value
    "Mean Jump Size": [0.1],  # Default value 
    "Jump Size Volatility": [0.2],  # Default value
    "Correlation": [-0.5]  # Default value
}

input_df_bns = pd.DataFrame(input_data_bns)
st.table(input_df_bns)

# Initialize BNS model
bns_model = BNSModel(
    initial_stock_price=current_price,
    strike_price=strike,
    time_to_maturity=time_to_maturity,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    jump_intensity=0.5,
    mean_jump_size=0.1,
    jump_size_volatility=0.2,
    correlation=-0.5
)

# Simulate price paths
simulated_paths = bns_model.simulate_paths(number_of_simulations, number_of_steps)

# Create plot
plt.figure(figsize=(10, 6))

# Plot each simulation path
for i in range(simulated_paths.shape[1]):
    plt.plot(simulated_paths[:, i], lw=1, alpha=0.6)

plt.title(f"BNS Model Simulation for {stock_symbol}")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.grid(True)
plt.axhline(y=strike, color='r', linestyle='--', label='Strike Price')
plt.legend()

# Display plot in Streamlit
st.pyplot(plt)



st.title("Binomial Pricing Model")

# Table of Inputs for Binomial Model
input_data_binomial = {
    "Stock Symbol": [stock_symbol],
    "Strike Price": [strike], 
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [risk_free_rate],
    "Number of Steps": [number_of_steps]
}

input_df_binomial = pd.DataFrame(input_data_binomial)
st.table(input_df_binomial)

# Calculate option prices using binomial model
binomial_model = BinomialOptionPricingModel(
    stock_symbol=stock_symbol,
    time_to_maturity=time_to_maturity,
    strike=strike,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    num_simulations=number_of_simulations,
    num_steps=number_of_steps
)

# Generate simulated prices if needed
simulated_prices = binomial_model.simulate_stock_price()  # Ensure this method exists

# Calculate option prices
call_price_binomial = binomial_model.calculate_option_price(simulated_prices)  # Pass simulated prices
put_price_binomial = binomial_model.calculate_option_price(simulated_prices)  # Pass simulated prices

# Display results in a table
results_data = {
    "Option Type": ["Call Option", "Put Option"],
    "Price": [f"${call_price_binomial:.2f}", f"${put_price_binomial:.2f}"]
}

results_df = pd.DataFrame(results_data)
st.table(results_df)

# Remove the plot_tree call if not needed
# fig = binomial_model.plot_tree()  # This line can be removed
# st.pyplot(fig)

st.title("Monte Carlo Pricing Model")
# Table of Inputs
input_data_monte_carlo = {
    "Stock Symbol" : [stock_symbol], 
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [risk_free_rate],
    "Number of Simulations" : [number_of_simulations], 
    "Number of Steps": [number_of_steps]
}

input_df_monte_carlo = pd.DataFrame(input_data_monte_carlo)
st.table(input_df_monte_carlo)

mc_option = MonteCarloOptionPricingModel(stock_symbol, time_to_maturity, strike, risk_free_rate, volatility, number_of_simulations, number_of_steps)
simulated_prices = mc_option.simulate_stock_price()
plt = mc_option.plot_simulations(simulated_prices)
st.pyplot(plt)

