import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import wrap

# _______________________________________________________________________________________________________ #
# ________________________ Plotting Tool Calling and Topic Adherence Evaluations ________________________ #
# _______________________________________________________________________________________________________ #
def plot_all_tools_called_eval(response: dict, agent_internal_state: str):
    """
    Creates a visualization of tool usage evaluation results for a specific agent.
    
    This function generates a multi-subplot figure where each subplot represents an iteration
    of tool usage. Each plot shows:
    - Bar chart of tool call counts
    - Error markers and annotations for failed calls
    - Timestamp of iteration
    - Overall tools usage status
    
    Args:
        response (dict): The complete state dictionary from the agent run containing evaluation results.
                        Expected structure:
                        {
                            agent_name: {
                                'all_tools_eval': {
                                    'stats': List[Dict],  # List of iteration statistics
                                    'passed': List[bool]  # List of pass/fail results
                                }
                            }
                        }
        agent_name (str): Name of the agent whose results should be plotted
                         (e.g., 'news_sentiment_agent_internal_state')
    
    Returns:
        None: Displays the plot using plt.show()
    """
     
     # Check if agent's internal state is empty
    all_tools_eval_results = response[agent_internal_state]['all_tools_eval']
    if not all_tools_eval_results['stats']:
        print(f"\n⚠️ No tool evaluation data available for {agent_internal_state}")
        return
    
     # response contains the output state of te Fin agent after a run
    all_tools_eval_results =response[agent_internal_state]['all_tools_eval']

    # Calculate number of iterations and set up the subplots
    n_iterations = len(all_tools_eval_results['stats'])
    fig, axes = plt.subplots(n_iterations, 1, figsize=(12, 5*n_iterations))
    if n_iterations == 1:
        axes = [axes]  # Convert to list for consistent indexing

    # Process each iteration
    for iteration_idx, (ax, iteration_stats) in enumerate(zip(axes, all_tools_eval_results['stats'])):
        # Create DataFrame for this iteration
        tool_counts = iteration_stats['tool_counts']
        
        # Create a mapping of tools to their error messages
        error_messages = {}
        for error in iteration_stats['errors']['execution_errors']:
            tool = error['tool']
            error_msg = error['error'].get('error', str(error['error']))
            error_messages[tool] = error_msg
        
        # Create DataFrame for this iteration
        iteration_data = []
        for tool, count in tool_counts.items():
            has_error = tool in error_messages
            iteration_data.append({
                'Tool': tool,
                'Calls': count,
                'Has Error': has_error,
                'Error Message': error_messages.get(tool, 'No error')
            })
        
        df = pd.DataFrame(iteration_data)
        
        # Create bar plot for this iteration
        sns.barplot(
            data=df,
            x='Tool',
            y='Calls',
            ax=ax,
            color='skyblue'
        )
        
        # Add error markers and annotations
        for idx, row in df.iterrows():
            if row['Has Error']:
                # Plot red X
                ax.plot(
                    idx,
                    row['Calls'],
                    'rx',
                    markersize=10,
                    markeredgewidth=2,
                    label='Error'
                )
                
                # Add error message annotation
                ax.annotate(
                    f"Error: {row['Error Message']}",
                    xy=(idx, row['Calls']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=8,
                    wrap=True
                )
            
            # Add annotation for zero calls
            if row['Calls'] == 0:
                ax.annotate(
                    'Not Called',
                    xy=(idx, 0),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color='red',
                    fontsize=8
                )
        
        # Customize subplot
        ax.set_title(f'Iteration {iteration_idx + 1}', pad=20, size=12)
        ax.set_xlabel('Tools', labelpad=10)
        ax.set_ylabel('Number of Calls', labelpad=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Add timestamp
        timestamp = iteration_stats['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.98, 0.98, f'Timestamp: {timestamp}', 
                transform=ax.transAxes, 
                ha='right', 
                va='top', 
                fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add all_tools_used status
        status_text = "✓ All tools used" if iteration_stats['all_tools_used'] else "✗ Not all tools used"
        ax.text(0.02, 0.98, status_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8,
                color='green' if iteration_stats['all_tools_used'] else 'red',
                bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return 

def plot_topic_adhrence_eval( response: dict , agent_internal_state = str) :


    """
    Creates a visualization of topic adherence evaluation results for a specific agent.
    
    This function generates a single figure showing the topic adherence evaluation results
    across all iterations. Each iteration is displayed with:
    - Color-coded background (green for pass, red for fail)
    - Iteration number
    - Evaluation status
    - Detailed reason for the evaluation
    
    Args:
        response (dict): The complete state dictionary from the agent run containing evaluation results.
                        Expected structure:
                        {
                            agent_name: {
                                'topic_adherence_eval': {
                                    'passed': List[str],  # List of "true"/"false" strings
                                    'reason': List[str]   # List of evaluation reasons
                                }
                            }
                        }
        agent_name (str): Name of the agent whose results should be plotted
                         (e.g., 'news_sentiment_agent_internal_state')
    
    Returns:
        None: Displays the plot using plt.show()
    """

     # Check if agent's internal state is empty
    topic_adherence_eval_results = response[agent_internal_state]['topic_adherence_eval']
    if not topic_adherence_eval_results['passed']:
        print(f"\n⚠️ No topic adherence evaluation data available for {agent_internal_state}")
        return
    
     # response contains the output state of te Fin agent after a run
    topic_adherence_eval_results =response[agent_internal_state]['topic_adherence_eval']
  
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create data for visualization
    iterations = len(topic_adherence_eval_results['passed'])
    y_positions = range(iterations)

    # Add background colors for each iteration (adjusted height)
    for i, status in enumerate(topic_adherence_eval_results['passed']):
        color = '#ff9999' if status.lower() == 'false' else '#99ff99'
        # Extend the background to fully include the iteration label
        ax.axhspan(i-0.3, i+0.3, color=color, alpha=0.3)

    # Add iteration numbers and evaluation details
    for i, (status, reason) in enumerate(zip(topic_adherence_eval_results['passed'], 
                                        topic_adherence_eval_results['reason'])):
        # Move iteration number inside the colored background
        ax.text(0.02, i, f'Iteration {i+1}:', 
                ha='left', va='center',
                fontsize=11, fontweight='bold')
        
        # Format the reason text (wrap at 70 characters)
        wrapped_reason = '\n'.join(wrap(reason, width=70))
        
        # Create display text with status and reason (moved right to accommodate iteration number)
        display_text = f"Status: {status.upper()}\n{wrapped_reason}"
        
        ax.text(0.15, i, display_text,
                ha='left', va='center',
                fontsize=10,
                color='darkred' if status.lower() == 'false' else 'darkgreen')

    # Customize plot
    ax.set_title('Topic Adherence Evaluation', pad=20, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, iterations-0.5)

    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add a light grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return

# _______________________________________________________________________________________________________ #
# ________________________ Rendering Tables for Tool Calling and Topic Adherence Evaluations ________________________ #
# _______________________________________________________________________________________________________ #
def get_all_tools_called_eval_df(response: dict, agent_internal_state: str) -> pd.DataFrame:
    """
    Returns a DataFrame of tool usage evaluation results for a specific agent.
    
    Args:
        response (dict): The complete state dictionary from the agent run containing evaluation results.
        agent_internal_state (str): Name of the agent whose results should be processed.
    
    Returns:
        pd.DataFrame: DataFrame containing tool usage evaluation results.
    """
    all_tools_eval_results = response[agent_internal_state]['all_tools_eval']
    if not all_tools_eval_results['stats']:
        print(f"\n⚠️ No tool evaluation data available for {agent_internal_state}")
        return pd.DataFrame()  # Return empty DataFrame if no data

    # Collect data for each iteration
    data = []
    for iteration_idx, iteration_stats in enumerate(all_tools_eval_results['stats']):
        tool_counts = iteration_stats['tool_counts']
        error_messages = {error['tool']: error['error'].get('error', str(error['error']))
                          for error in iteration_stats['errors']['execution_errors']}
        
        for tool, count in tool_counts.items():
            data.append({
                'Iteration': iteration_idx + 1,
                'Tool': tool,
                'Calls': count,
                'Has Error': tool in error_messages,
                'Error Message': error_messages.get(tool, 'No error'),
                'Timestamp': iteration_stats['timestamp'],
                'All Tools Used': iteration_stats['all_tools_used']
            })
    
    return pd.DataFrame(data)

def get_topic_adherence_eval_df(response: dict, agent_internal_state: str) -> pd.DataFrame:
    """
    Returns a DataFrame of topic adherence evaluation results for a specific agent.
    
    Args:
        response (dict): The complete state dictionary from the agent run containing evaluation results.
        agent_internal_state (str): Name of the agent whose results should be processed.
    
    Returns:
        pd.DataFrame: DataFrame containing topic adherence evaluation results.
    """
    topic_adherence_eval_results = response[agent_internal_state]['topic_adherence_eval']
    if not topic_adherence_eval_results['passed']:
        print(f"\n⚠️ No topic adherence evaluation data available for {agent_internal_state}")
        return pd.DataFrame()  # Return empty DataFrame if no data

    # Collect data for each iteration
    data = []
    for i, (status, reason) in enumerate(zip(topic_adherence_eval_results['passed'], 
                                             topic_adherence_eval_results['reason'])):
        data.append({
            'Iteration': i + 1,
            'Status': status,
            'Reason': reason
        })
    
    return pd.DataFrame(data)
# _______________________________________________________________________________________________________ #
# ________________________ Plotting Evaluations for SQL Agent ___________________________________________ #
# _______________________________________________________________________________________________________ #

def plot_sql_agent_errors(error_list, title, ax):
    """
    Create a visualization of SQL Agent execution errors.
    
    Args:
        error_list (list): List of dictionaries containing error information
        title (str): Title for the plot
        ax: Matplotlib axis to plot on
    """
    # Modern color palette - using blues and purples
    colors = ['#7986CB', '#9575CD', '#64B5F6', '#4DB6AC', '#7E57C2']
    if len(error_list) > len(colors):
        colors = colors * (len(error_list) // len(colors) + 1)
    
    # Create the horizontal bar plot
    y = np.arange(len(error_list))
    bars = ax.barh(y, [1] * len(error_list), 
                  color=colors[:len(error_list)], 
                  alpha=0.8,
                  height=0.7)
    
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_ylabel("Iteration", fontsize=12, color='#2c3e50')
    
    # Set integer ticks for y-axis
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{i+1}" for i in y], color='#2c3e50')
    
    # Remove x-axis as we're only showing occurrence
    ax.set_xticks([])
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add error messages as centered text in bars
    for idx, error in enumerate(error_list):
        error_msg = error.get("Error message", "Unknown error")
        
        # Add error message text centered in bar
        ax.text(0.5, idx, error_msg,
                ha='center',
                va='center',
                fontsize=10,
                color='black',
                fontweight='medium',
                transform=ax.get_yaxis_transform())
    
    # Add subtle grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.2, color='#7f8c8d')
    ax.set_axisbelow(True)

def visualize_sql_agent_performance(**error_lists):
    """
    Visualize SQL Agent errors with flexible input handling.
    
    Args:
        **error_lists: Dictionary of error lists with their titles
        Example: visualize_sql_agent_performance(
            query_errors={"data": query_list, "title": "Query Generation Errors"},
            format_errors={"data": format_list, "title": "Result Formatting Errors"}
        )
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out empty lists while preserving titles
    active_lists = {
        k: v for k, v in error_lists.items() 
        if v["data"]
    }
    
    # If no errors in any list
    if not active_lists:
        plt.figure(figsize=(10, 5), facecolor='white')
        plt.text(0.5, 0.5, "No issues occurred during SQL Agent execution! 🎉", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                color='#2c3e50')
        plt.axis('off')
        plt.title("SQL Agent Execution Summary", 
                 pad=20, 
                 fontsize=16, 
                 color='#2c3e50',
                 fontweight='bold')
    
    # If only one list has content
    elif len(active_lists) == 1:
        plt.figure(figsize=(15, 5), facecolor='white')
        error_data = list(active_lists.values())[0]
        plot_sql_agent_errors(error_data["data"], error_data["title"], plt.gca())
    
    # If multiple lists have content
    else:
        fig, axes = plt.subplots(len(active_lists), 1, 
                                figsize=(15, 5*len(active_lists)), 
                                facecolor='white')
        if len(active_lists) == 1:
            axes = [axes]
        
        for ax, (_, error_data) in zip(axes, active_lists.items()):
            plot_sql_agent_errors(error_data["data"], error_data["title"], ax)
        
        plt.subplots_adjust(hspace=0.3)
    
    plt.tight_layout()
    plt.show()

# _______________________________________________________________________________________________________ #
# ________________________ Rendering Tables for Evaluations for SQL Agent _______________________________ #
# _______________________________________________________________________________________________________ #

def format_sql_agent_errors_table(error_list, title) -> pd.DataFrame:
    """
    Create a table representation of SQL Agent execution errors.
    
    Args:
        error_list (list): List of dictionaries containing error information
        title (str): Title for the table
    
    Returns:
        pd.DataFrame: Formatted table of errors
    """
    # Create DataFrame with iteration numbers and error messages
    df = pd.DataFrame([
        {
            'Iteration': f"#{i+1}",
            'Error Message': error.get("Error message", "Unknown error")
        }
        for i, error in enumerate(error_list)
    ])
    
    return df

def tabulate_sql_agent_performance(**error_lists) -> pd.DataFrame:

    """
    Create tables for SQL Agent errors with flexible input handling.
    
    Args:
        **error_lists: Dictionary of error lists with their titles
        Example: tabulate_sql_agent_performance(
            query_errors={"data": query_list, "title": "Query Generation Errors"},
            format_errors={"data": format_list, "title": "Result Formatting Errors"}
        )
    
    Returns:
        pd.DataFrame: Combined DataFrame containing all errors or success message
    """
    # Filter out empty lists while preserving titles
    active_lists = {
        k: v for k, v in error_lists.items() 
        if v["data"]
    }
    
    # If no errors in any list
    if not active_lists:
        return pd.DataFrame({
            'Category': ['SQL Agent Execution'],
            'Status': ['Success'],
            'Details': ['No issues occurred during execution']
        })
    
    # Create tables for each error type and combine them
    all_errors = []
    for error_type, error_data in active_lists.items():
        errors_df = format_sql_agent_errors_table(error_data["data"], error_data["title"])
        errors_df['Category'] = error_data["title"]  # Add category column
        all_errors.append(errors_df)
    
    # Combine all error DataFrames
    return pd.concat(all_errors, ignore_index=True)


# Example usage:
# result_table = tabulate_sql_agent_performance(
#     query_errors={"data": query_list, "title": "Query Generation Errors"},
#     format_errors={"data": format_list, "title": "Result Formatting Errors"}
# )

# # Display the table
# print(result_table.to_string(index=False))