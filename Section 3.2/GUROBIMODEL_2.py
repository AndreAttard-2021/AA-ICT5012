# Importing gurobipy to utilise Gurobi Optimizer
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

# Importing numpy to handle arrays
import numpy as np
from numpy import genfromtxt
# Importing pandas to handle dataframes
import pandas as pd

# Importing datetime module to interact with date/time instances
from datetime import datetime

# Importing math package to utilise mathematical functions
import math
# Importing ast package to process trees of the Python abstract syntax grammar
import ast
# Importing re package to utilise regular expression operations
import re
import argparse

def run_gurobi_model(
    filename,
    Bus_Terminals,
    L=42,
    q=11,
    m_max=15,
    time_limit=1200,
    seed=None
):
    
    # Create an environment with your WLS license
    params = {
    "WLSACCESSID": '0494f3c5-52dc-4366-b02e-19b191063953',
    "WLSSECRET": '07772747-f91c-4066-a5ea-11ec93ecc22b',
    "LICENSEID": 2604445,
    }
    env = gp.Env(params=params)
    
    
    # Function 3 - Obtain Dictionary of Routes
    def Routes(Routes_DirectionsWNumber, m_max):
    
        """
        Connects entries in 'Routes_DirectionsWNumber'.
        Recall 'Routes_DirectionsWNumber' is given in format [[i,j,c],...] where i represents origin bus stop,
        j represents destination bus stop and c represents route number
        Hence final result will be:
        'Route c': [i,j],[j,k],[k,f],... where k and f are two other nodes present in route c
        This is repeated for any distinct c
    
        Parameters:
        Routes_DirectionsWNumber: The routes considered in a list with format [[i,j,c],...] where i represents origin bus stop,
        j represents destination bus stop and c represents route number
        m_max: The maximum number of routes to be considered
    
        Returns:
        routes_ordered_complete (dict): A complete representation of the path taken by each route.
        Note that the direction of non-circular routes are distinguished from each other by appending '_1' to route number.
        """
        # Initialising empty lists for 'routes', 'routes_ordered' and 'routes_ordered_complete'
        routes = {}
        routes_ordered = {}
        routes_ordered_complete = {}
    
        # for loop going over maximum number of routes to be considered
        # Not that the maximum number of routes is not necessarily utilised in the final solution
        for j in range(1, m_max+1):
            # for each route number 'j' being considered, an empty list within a dictionary 'routes' is initialised
            routes[f'Route_{j}'] = []
            # for loop going over each entry in 'Routes_DirectionsWNumber'
            for i in Routes_DirectionsWNumber:
                # if entry 'i' has route number equivalent to route number 'j' being considered then
                # origin-destination pair of entry 'i' are added to route list.
                if i[2] == j:
                    routes[f'Route_{j}'].append([i[0],i[1]])
    
        # for loop going over each route in the 'routes' dictionairy defined above
        for i in routes:
            # for each route 'i' being considered, an empty list within a dictionary 'routes_ordered' is initialised
            routes_ordered[f'{i}'] = []
            # Route_Direction variable is defined which is equivalent to a list of all entries in the route 'i' being considered
            Route_Direction = routes[i][:]
            # while there exists 'Route_Direction'
            while Route_Direction:
                # taking each entry 'j' in 'Route_Direction' equivalent to [i,j,c] where i represents origin bus stop,
                # j represents destination bus stop and c represents route 'i' being considered
                for j in Route_Direction[:]:
                    # first entry must always be a bus terminal, hence once an entry whose origin bus stop is presen in 'Bus_Terminals'
                    # is found then this is considered to be the initial origin-destination pair of route
                    if j[0] in Bus_Terminals:
                        # append origin node of entry 'j' to 'routes_ordered' dictionary for route 'i'
                        routes_ordered[f'{i}'].append(j[0])
                        # append destination node of entry 'j' to 'routes_ordered' dictionary for route 'i'
                        routes_ordered[f'{i}'].append(j[1])
                        # remove entry 'j' from 'Route_Direction' list
                        Route_Direction.remove(j)
                    # for any other entry of 'j' it is appended to 'routes_ordered' dictionary for route 'i' if and only if:
                    # 1) the origin node of entry 'j' is equivalent to last node present in 'routes_ordered' dictionary for route 'i'
                    # 2) the length of 'routes_ordered' dictionary for route 'i' is greater than zero (Thus it is not the first entry
                    # which must always be a Bus_Terminal)
                    elif len(routes_ordered[f'{i}']) > 0 and j[0] == routes_ordered[f'{i}'][-1]:
                        # append destination node of entry 'j' to 'routes_ordered' dictionary for route 'i'
                        routes_ordered[f'{i}'].append(j[1])
                        # remove entry 'j' from 'Route_Direction' list
                        Route_Direction.remove(j)
                    else:
                        pass
    
        # Recall that if a route starts and ends with different terminals, then route also repeats itself in opposite direction
        # for loop going over all routes 'i' in routes_ordered' dictionary so that a route going in the opposite direction can
        # be obtained for any non-circular route
        for i in routes_ordered:
            # if len of route 'i' in 'routes_ordered' dictionary is greater than zero and
            # first and last stop of route are equivalent, then this is a circular route.
            if len(routes_ordered[i]) > 0 and routes_ordered[i][0] == routes_ordered[i][-1]:
                # Copy route 'i' in 'routes_ordered' dictionary as route 'i' in 'routes_ordered_complete'
                routes_ordered_complete[f'{i}'] = routes_ordered[i]
            # if len of route 'i' in 'routes_ordered' dictionary is greater than zero and
            # first and last stop of route are not equivalent, then this is a non-circular route.
            elif len(routes_ordered[i]) > 0 and routes_ordered[i][0] != routes_ordered[i][-1]:
                # Copy route 'i' in 'routes_ordered' dictionary as route 'i' in 'routes_ordered_complete' dictionary
                routes_ordered_complete[f'{i}'] = routes_ordered[i]
                # Copy route 'i' in 'routes_ordered' dictionary going in reverse order as route 'i_1' in 'routes_ordered_complete' dictionary
                routes_ordered_complete[f'{i}_1'] = routes_ordered[i][::-1]
            else:
                # if none of the above conditions are met an empty list is denoted as route 'i' in 'routes_ordered_complete' dictionary
                routes_ordered_complete[f'{i}'] = []
    
        return routes_ordered_complete
    
    
    # 1. Define the Floyd–Warshall function
    def floyd_warshall(cost_matrix):
        """
        Runs the Floyd-Warshall algorithm on the given cost matrix.
    
        :param cost_matrix: A 2D list (NxN) of direct travel times.
                            cost_matrix[i][j] = float('inf') if no direct path.
        :return: A 2D list (NxN) where the value at [i][j] is the minimum travel time
                 from node i to node j.
        """
        n = len(cost_matrix)
    
        # Initialize the distance matrix as a copy of the original cost matrix
        dist = [[cost_matrix[i][j]*60 for j in range(n)] for i in range(n)]
    
        # Run the Floyd–Warshall updates
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    
        return dist
            
    # ---------------------------
    # Data and Model Initialization
    # ---------------------------
    model1 = gp.Model(env=env)
    if seed is not None:
        model1.Params.Seed = seed
    
    #Obtaining Maximum Drive Time required to normalize our objective function
    filename = filename
    raw_matrix = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Split each line by whitespace.
            # If your file uses commas, use: line.strip().split(',')
            tokens = line.strip().split()
            raw_matrix.append(tokens)
    
    # 3. Convert string "Inf" to float('inf') and other strings to floats
    cost_matrix = []
    for row in raw_matrix:
        new_row = []
        for token in row:
            if token.strip().lower() == "inf":
                new_row.append(float('inf'))
            else:
                new_row.append(float(token))
        cost_matrix.append(new_row)
    
    cost_matrix = [row for row in cost_matrix if any(row)]  # Removes empty lists
    
    # 4. Run the Floyd–Warshall algorithm
    drive_times_matrix = floyd_warshall(cost_matrix)
    max_value = max(max(row) for row in drive_times_matrix)
    
    
    
    
    
    # Travel Time Matrix
    t_inv = genfromtxt(filename, delimiter=None)
    finite_mask = np.isfinite(t_inv)
    # Replace any 'inf' instances with '5000000'
    #t_inv[~finite_mask] = 5000000
    
    # Number of Bus Stops
    n = len(t_inv)
    
    # Bus Terminals and Normal Stops
    Bus_Terminals = Bus_Terminals
    all_stops = list(range(n))
    Normal_Stops = [i for i in all_stops if i not in Bus_Terminals]
    
    # Route parameters
    L = L      # Maximum number of stops per route
    q = q      # Minimum number of stops per route
    m_max = m_max # Maximum number of routes
    
    # ---------------------------
    # Decision Variables
    # ---------------------------
    
    # x[i,j,c] = 1 if arc (i,j) is used in route c
    x = {}
    for i in range(n):
        for j in range(n):
            if not finite_mask[i, j]:
                continue  # Skip arcs with infinite travel time
            for c in range(1, m_max+1):
                x[i, j, c] = model1.addVar(vtype=GRB.BINARY, name=f'x{i},{j},{c}')
    
    # u[i,c] tracks the sequence in which node i is visited in route c (for sub-tour elimination)
    u = model1.addVars(n, m_max+1, vtype=GRB.INTEGER, lb=1, ub=n-1, name="u")
    
    # y[c] = 1 if route c is used
    y = model1.addVars(range(1, m_max+1), vtype=GRB.BINARY, name="y")
    
    # v[i] = 1 if node i is visited on any route
    v = {}
    for i in range(n):
        v[i] = model1.addVar(vtype=GRB.BINARY, name=f"v_{i}")
    
    # z[i,c] = 1 if node i is visited in route c
    z = model1.addVars(range(n), range(1, m_max+1), vtype=GRB.BINARY, name="z")
    
    # ---------------------------
    # Objective Function
    # ---------------------------
    cost_travel = gp.quicksum(
        t_inv[i][j] * x[i, j, c]
        for i in range(n) for j in range(n) for c in range(1, m_max+1)
        if (i, j, c) in x  # ensure variable exists
    )
    cost_travel_normalised = cost_travel / (m_max * max_value)
    
    Num_Dist_Nodes = gp.quicksum(v[i] for i in range(n))
    Num_Dist_Nodes_Normalised = 5 * ((n - Num_Dist_Nodes) / n)
    
    model1.setObjective(cost_travel_normalised + Num_Dist_Nodes_Normalised, GRB.MINIMIZE)
    
    # ---------------------------
    # Constraints
    # ---------------------------
    
    # Constraint Set 1 - Restricting routes to start and end at Bus Terminals (Constraint Set 1)
    for c in range(1, m_max+1):
        # Start at Bus Terminal: sum over arcs leaving a bus terminal
        model1.addConstr(
            gp.quicksum(x[i, j, c] for i in Bus_Terminals for j in range(n) if (i, j, c) in x) <= 1,
            name=f"start_terminal_route_{c}"
        )
        # End at Bus Terminal: sum over arcs entering a bus terminal
        model1.addConstr(
            gp.quicksum(x[i, j, c] for j in Bus_Terminals for i in range(n) if (i, j, c) in x) <= 1,
            name=f"end_terminal_route_{c}"
        )
    
    # Constraint Set 2 - Keep track of all visited nodes
    for i in range(n):
        expr = gp.quicksum(x[i, j, c] for c in range(1, m_max+1) for j in range(n) if (i, j, c) in x) + \
               gp.quicksum(x[j, i, c] for c in range(1, m_max+1) for j in range(n) if (j, i, c) in x)
        model1.addConstr(expr >= v[i], name=f"node_{i}_visited_lower")
    
    # Constraint Set 3 - Ensure cohesive routes for Normal Stops (Constraint Set 3)
    for s in Normal_Stops:
        for c in range(1, m_max+1):
            model1.addConstr(
                gp.quicksum(x[i, s, c] for i in range(n) if (i, s, c) in x) -
                gp.quicksum(x[s, j, c] for j in range(n) if (s, j, c) in x) == 0,
                name=f"cohesive_normal_stop_{s}_route_{c}"
            )
    
    # Constraint Set 4 - Bus Terminal usage: used twice per route (once to start and once to end) (Constraint Set 2)
    for c in range(1, m_max+1):
        model1.addConstr(
            gp.quicksum(x[i, j, c] for i in Bus_Terminals for j in range(n) if (i, j, c) in x) -
            gp.quicksum(x[i, j, c] for j in Bus_Terminals for i in range(n) if (i, j, c) in x) == 0,
            name=f"balance_terminal_route_{c}"
        )
    
    # Constraint Set 5.1 - No self loops (Constraint Set 4)
    for i in range(n):
        for c in range(1, m_max+1):
            if (i, i, c) in x:
                model1.addConstr(x[i, i, c] == 0, name=f"no_self_loop_{i}_route_{c}")
    
    # Constraint Set 5.2 - A stop cannot be visited more than once in the same route
    for i in range(n):
        for c in range(1, m_max+1):
            expr = gp.quicksum(x[i, j, c] for j in range(n) if (i, j, c) in x and j != i) + \
                   gp.quicksum(x[j, i, c] for j in range(n) if (j, i, c) in x and j != i)
            model1.addConstr(expr <= 2, name=f"visit_once_{i}_route_{c}")
    
    # Constraint Set 6 - Sub-tour elimination (Kulkarni-Bhave SECs)
    for c in range(1, m_max+1):
        for idx_i in range(len(Normal_Stops)):
            for idx_j in range(len(Normal_Stops)):
                if idx_i != idx_j:
                    i_node = Normal_Stops[idx_i]
                    j_node = Normal_Stops[idx_j]
                    # Only add constraint if the variables x[i,j,c] exist (or default to 0 if missing)
                    model1.addConstr(
                        u[i_node, c] - u[j_node, c] +
                        L * (x[i_node, j_node, c] if (i_node, j_node, c) in x else 0) +
                        (L - 2) * (x[j_node, i_node, c] if (j_node, i_node, c) in x else 0)
                        <= L - 1,
                        name=f"subtour_route_{c}_{i_node}_{j_node}"
                    )
    
    # Constraint Set 7 - Ensure each route visits at least q stops and at most L stops
    for i in range(n):
        for c in range(1, m_max+1):
            model1.addConstr(
                gp.quicksum(x[i, j, c] for j in range(n) if (i, j, c) in x) +
                gp.quicksum(x[j, i, c] for j in range(n) if (j, i, c) in x)
                >= z[i, c],
                name=f"link_z_{i}_{c}"
            )
    for c in range(1, m_max+1):
        model1.addConstr(
            gp.quicksum(z[i, c] for i in range(n)) >= q,
            name=f"min_stops_route_{c}"
        )
        model1.addConstr(
            gp.quicksum(z[i, c] for i in range(n)) <= L,
            name=f"max_stops_route_{c}"
        )
    
    # (Optional) Constraint Set 8 - Linking route usage y[c] with arc usage in route c (using Big-M or similar techniques)
    # This section is commented out. It can be added back if needed.
    # M = 10000
    # epsilon = 1e-6
    # for c in range(1, m_max+1):
    #     model1.addConstr(
    #         gp.quicksum(x[i, j, c] for i in range(n) for j in range(n) if (i,j,c) in x) >= y[c] * epsilon,
    #         name=f"RouteUsed_{c}_Positive"
    #     )
    #     model1.addConstr(
    #         gp.quicksum(x[i, j, c] for i in range(n) for j in range(n) if (i,j,c) in x) <= y[c] * M,
    #         name=f"RouteUsed_{c}_Negative"
    #     )
    
    # Set Time Limit for Gurobi
    model1.Params.TimeLimit = time_limit
    
    # ---------------------------
    # Optimization and Output
    # ---------------------------
    model1.optimize()
    
    # Build a list of routes from the solution
    if model1.SolCount > 0:
    
        Routes_DirectionsWNumber = []
        for var in model1.getVars():
            if var.X > 0.5:
                parts = var.varName.split(',')
                try:
                    i = int(parts[0][1:])  # Remove leading 'x'
                    j = int(parts[1])
                    c = int(parts[2])
                    Routes_DirectionsWNumber.append([i, j, c])
                except (IndexError, ValueError):
                    pass
    
        print("Objective Function at this solution:", Routes_DirectionsWNumber)
        print("List of All Routes is", Routes(Routes_DirectionsWNumber, m_max))
    
    else:
        print("No feasible solution found.")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Gurobi bus-route model')
    parser.add_argument('--filename', type=str, required=True,
                        help='Path to travel times file')
    parser.add_argument('--terminals', type=str, required=True,
                        help='Python list of bus terminal indices, e.g. "[0,10,12]"')
    parser.add_argument('--L', type=int, default=42,
                        help='Max stops per route')
    parser.add_argument('--q', type=int, default=11,
                        help='Min stops per route')
    parser.add_argument('--m_max', type=int, default=15,
                        help='Max number of routes')
    parser.add_argument('--time_limit', type=int, default=1200,
                        help='Solver time limit in seconds')
    parser.add_argument('--seed', type=int, help='Random seed for Gurobi')
    args = parser.parse_args()

    # parse terminals list
    Bus_Terminals = ast.literal_eval(args.terminals)

    run_gurobi_model(
        filename=args.filename,
        Bus_Terminals=Bus_Terminals,
        L=args.L,
        q=args.q,
        m_max=args.m_max,
        time_limit=args.time_limit,
        seed=args.seed
    )

def run_gurobi_model(
    filename,
    Bus_Terminals,
    L=42,
    q=11,
    m_max=15,
    time_limit=1200,
    seed=None,
    wls_params=None
):
    """
    Execute the Gurobi model and return objective + routes.

    Parameters:
    - filename: path to travel times file
    - Bus_Terminals: list of terminal node indices
    - L: max stops per route
    - q: min stops per route
    - m_max: max number of routes
    - time_limit: solver time limit in seconds
    - seed: random seed for reproducibility
    - wls_params: optional Gurobi license params
    """
    env = _create_env(wls_params)
    model = gp.Model(env=env)
    if seed is not None:
        model.Params.Seed = seed
    
    # Load and process travel-time matrix
    raw = []
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            raw.append(tokens)
    matrix = [[float('inf') if t.lower()=='inf' else float(t) for t in row] for row in raw if row]

    # Compute all-pairs shortest paths
    dist = floyd_warshall(matrix)
    max_val = max(max(row) for row in dist)

    # Inverse travel time array
    t_inv = genfromtxt(filename, delimiter=None)
    finite = np.isfinite(t_inv)
    n = len(t_inv)
    nodes = list(range(n))
    Normal_Stops = [i for i in nodes if i not in Bus_Terminals]

    # ---------------------------
    # Decision Variables
    # ---------------------------
    # x[i,j,c] = 1 if arc (i,j) is used in route c
    x = {}
    for i in nodes:
        for j in nodes:
            if not finite[i, j]: continue  # Skip infinite travel-times
            for c in range(1, m_max+1):
                x[i, j, c] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{c}')

    # u[i,c] for sub-tour elimination
    u = model.addVars(n, range(1, m_max+1), vtype=GRB.INTEGER, lb=1, ub=n-1, name='u')
    # y[c] = 1 if route c is used
    y = model.addVars(range(1, m_max+1), vtype=GRB.BINARY, name='y')
    # v[i] = 1 if node i is visited on any route
    v = {i: model.addVar(vtype=GRB.BINARY, name=f'v_{i}') for i in nodes}
    # z[i,c] = 1 if node i is visited in route c
    z = model.addVars(nodes, range(1, m_max+1), vtype=GRB.BINARY, name='z')

    # ---------------------------
    # Objective Function
    # ---------------------------
    # Normalised travel cost
    travel_cost = gp.quicksum(t_inv[i][j] * x[i, j, c] for (i,j,c) in x)
    cost_travel_normalised = travel_cost / (m_max * max_val)
    # Maximise coverage of nodes (minimise unvisited)
    Num_Dist_Nodes = gp.quicksum(v[i] for i in nodes)
    Num_Dist_Nodes_normalised = 5 * ((n - Num_Dist_Nodes) / n)
    model.setObjective(cost_travel_normalised + Num_Dist_Nodes_normalised, GRB.MINIMIZE)

    # ---------------------------
    # Constraints
    # ---------------------------
    # Constraint Set 1 - Restrict routes to start and end at Bus Terminals
    for c in range(1, m_max+1):
        # Start at Bus Terminal
        model.addConstr(
            gp.quicksum(x[i, j, c] for i in Bus_Terminals for j in nodes if (i,j,c) in x) <= 1,
            name=f'start_terminal_route_{c}'
        )
        # End at Bus Terminal
        model.addConstr(
            gp.quicksum(x[i, j, c] for j in Bus_Terminals for i in nodes if (i,j,c) in x) <= 1,
            name=f'end_terminal_route_{c}'
        )

    # Constraint Set 2 - Keep track of all visited nodes
    for i in nodes:
        expr = gp.quicksum(x[i, j, c] for c in range(1, m_max+1) for j in nodes if (i,j,c) in x) + \
               gp.quicksum(x[j, i, c] for c in range(1, m_max+1) for j in nodes if (j,i,c) in x)
        model.addConstr(expr >= v[i], name=f'node_{i}_visited_lower')

    # Constraint Set 3 - Ensure cohesive routes for Normal Stops
    for s in Normal_Stops:
        for c in range(1, m_max+1):
            model.addConstr(
                gp.quicksum(x[i, s, c] for i in nodes if (i,s,c) in x) - \
                gp.quicksum(x[s, j, c] for j in nodes if (s,j,c) in x) == 0,
                name=f'cohesive_normal_stop_{s}_route_{c}'
            )

    # Constraint Set 4 - Bus Terminal usage: used once to start and once to end per route
    for c in range(1, m_max+1):
        model.addConstr(
            gp.quicksum(x[i, j, c] for i in Bus_Terminals for j in nodes if (i,j,c) in x) - \
            gp.quicksum(x[i, j, c] for j in Bus_Terminals for i in nodes if (i,j,c) in x) == 0,
            name=f'balance_terminal_route_{c}'
        )

    # Constraint Set 5.1 - No self loops
    for i in nodes:
        for c in range(1, m_max+1):
            if (i,i,c) in x:
                model.addConstr(x[i, i, c] == 0, name=f'no_self_loop_{i}_route_{c}')

    # Constraint Set 5.2 - A stop cannot be visited more than once in the same route
    for i in nodes:
        for c in range(1, m_max+1):
            expr = gp.quicksum(x[i, j, c] for j in nodes if (i,j,c) in x and j != i) + \
                   gp.quicksum(x[j, i, c] for j in nodes if (j,i,c) in x and j != i)
            model.addConstr(expr <= 2, name=f'visit_once_{i}_route_{c}')

    # Constraint Set 6 - Sub-tour elimination (Kulkarni-Bhave SECs)
    for c in range(1, m_max+1):
        for idx_i in range(len(Normal_Stops)):
            for idx_j in range(len(Normal_Stops)):
                if idx_i != idx_j:
                    i_node = Normal_Stops[idx_i]
                    j_node = Normal_Stops[idx_j]
                    model.addConstr(
                        u[i_node, c] - u[j_node, c] + \
                        L * (x[i_node, j_node, c] if (i_node,j_node,c) in x else 0) + \
                        (L - 2) * (x[j_node, i_node, c] if (j_node,i_node,c) in x else 0) <= L - 1,
                        name=f'subtour_route_{c}_{i_node}_{j_node}'
                    )

    # Constraint Set 7 - Ensure each route visits at least q stops and at most L stops
    for i in nodes:
        for c in range(1, m_max+1):
            model.addConstr(
                gp.quicksum(x[i, j, c] for j in nodes if (i,j,c) in x) + \
                gp.quicksum(x[j, i, c] for j in nodes if (j,i,c) in x) >= z[i, c],
                name=f'link_z_{i}_{c}'
            )
    for c in range(1, m_max+1):
        model.addConstr(
            gp.quicksum(z[i, c] for i in nodes) >= q,
            name=f'min_stops_route_{c}'
        )
        model.addConstr(
            gp.quicksum(z[i, c] for i in nodes) <= L,
            name=f'max_stops_route_{c}'
        )

    # (Optional) Constraint Set 8 - Linking route usage y[c] with arc usage (commented out)
    # M = 10000
    # epsilon = 1e-6
    # for c in range(1, m_max+1):
    #     model.addConstr(
    #         gp.quicksum(x[i, j, c] for i in nodes for j in nodes if (i,j,c) in x) >= y[c] * epsilon,
    #         name=f'RouteUsed_{c}_Positive'
    #     )
    #     model.addConstr(
    #         gp.quicksum(x[i, j, c] for i in nodes for j in nodes if (i,j,c) in x) <= y[c] * M,
    #         name=f'RouteUsed_{c}_Negative'
    #     )

    # Solve
    model.Params.TimeLimit = time_limit
    model.optimize()

    if model.SolCount == 0:
        print("No feasible solution found.")
        return

    # Extract solution arcs and reconstruct routes
    dirs = []
    for var in model.getVars():
        if var.X > 0.5 and var.varName.startswith('x_'):
            _, i, j, c = var.varName.split('_')
            dirs.append((int(i), int(j), int(c)))

    routes = reconstruct_routes(dirs, m_max, Bus_Terminals)
    print("Objective:", model.ObjVal)
    print("Routes:")
    for r, seq in routes.items():
        print(f"  {r}: {seq}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Gurobi bus-route model')
    parser.add_argument('--filename', type=str, required=True,
                        help='Path to travel times file')
    parser.add_argument('--terminals', type=str, required=True,
                        help='Python list of bus terminal indices, e.g. "[0,10,12]"')
    parser.add_argument('--L', type=int, default=42,
                        help='Max stops per route')
    parser.add_argument('--q', type=int, default=11,
                        help='Min stops per route')
    parser.add_argument('--m_max', type=int, default=15,
                        help='Max number of routes')
    parser.add_argument('--time_limit', type=int, default=1200,
                        help='Solver time limit in seconds')
    parser.add_argument('--seed', type=int, help='Random seed for Gurobi')
    args = parser.parse_args()

    # parse terminals list
    Bus_Terminals = ast.literal_eval(args.terminals)

    run_gurobi_model(
        filename=args.filename,
        Bus_Terminals=Bus_Terminals,
        L=args.L,
        q=args.q,
        m_max=args.m_max,
        time_limit=args.time_limit,
        seed=args.seed
    )