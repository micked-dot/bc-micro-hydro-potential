import streamlit as st
import pystac_client
import rasterio 
from rasterio.windows import from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import box, shape, mapping
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import contextily as cxt

# Page config
st.set_page_config(page_title="Micro-Hydro Analysis Tool", layout="wide")
st.title("Micro-Hydro Analysis Tool")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================
st.sidebar.header("Input Parameters")

# Grade range inputs
st.sidebar.subheader("Grade Range (%)")
grade_min = st.sidebar.number_input("Minimum Grade (%)", value=5.0, min_value=0.0, max_value=100.0)
grade_max = st.sidebar.number_input("Maximum Grade (%)", value=15.0, min_value=0.0, max_value=100.0)

# Bounding box inputs (EPSG:3979)
st.sidebar.subheader("Bounding Box (EPSG:3979)")
bbox_min_x = st.sidebar.number_input("Min X", value=-1941300.0)
bbox_max_x = st.sidebar.number_input("Max X", value=-1917200.0)
bbox_min_y = st.sidebar.number_input("Min Y", value=709700.0)
bbox_max_y = st.sidebar.number_input("Max Y", value=725400.0)

st.sidebar.subheader("Site Parameters")
flow_rate = st.sidebar.number_input("Flow Rate (m3/s)", value=0.087)
head = st.sidebar.number_input("Head (m)", value=50)
community_population = st.sidebar.number_input("Community Population (number of people)", value=197)

# Run button
run_analysis = st.sidebar.button("Run Analysis", key="run_button")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
if run_analysis:
    try:
        with st.spinner("Loading elevation data..."):
            bbox_3979 = (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y)
            bbox_3979_crs = "EPSG:3979"

            # Convert EPSG:3979 to EPSG:4326 for rasterio/stac search
            bbox_3979_geom = mapping(box(*bbox_3979))
            #st.write(f"Original bbox in EPSG:3979: {bbox_3979}")
            transformed_to_4326 = transform_geom("EPSG:3979", "EPSG:4326", bbox_3979_geom)
            bbox_4326_from_3979 = shape(transformed_to_4326).bounds
            bbox = list(bbox_4326_from_3979)
            bbox_crs = "EPSG:4326"
            #st.write(f"Transformed bbox in EPSG:4326: {bbox}")

            # Link to ccmeo datacube stac-api
            stac_root = "https://datacube.services.geo.ca/stac/api"
            catalog = pystac_client.Client.open(stac_root)

            search = catalog.search(
                collections=['mrdem-30'], 
                bbox=bbox,
            ) 

            # Get the link to the data asset for mrdem-30 dtm
            links = []
            for page in search.pages():
                for item in page:
                    links.append(item.assets['dtm'].href)

            if not links:
                st.error("No elevation data found for the specified bounding box.")
            else:
                # Read AOI from the first COG
                with rasterio.open(links[0]) as src:
                    #st.write(f"Dataset CRS: {src.crs}")
                    #st.write(f"Dataset bounds: {src.bounds}")

                    # Transform bbox to dataset CRS
                    geom4326 = mapping(box(*bbox))
                    transformed_geom = transform_geom(bbox_crs, src.crs, geom4326)
                    transformed_bbox = shape(transformed_geom).bounds
                    #st.write(f"Transformed bbox (dataset CRS): {transformed_bbox}")
                    
                    # Define the window to read the values
                    window = from_bounds(transformed_bbox[0], transformed_bbox[1], 
                                         transformed_bbox[2], transformed_bbox[3], 
                                         src.transform)
                    #st.write(f"Computed window: {window}")
                    
                    # Read value from file
                    rst = src.read(1, window=window)
                    #st.write(f"Read array shape: {rst.shape}")
                    
                    # Compute basic stats
                    # if rst.size:
                    #     try:
                    #         st.write(f"Data min/max: {np.nanmin(rst):.2f} / {np.nanmax(rst):.2f}")
                    #     except Exception:
                    #         st.write("Could not compute min/max (array may be masked)")
                    
                    # Calculate slope/grade
                    pixel_size = src.transform.a
                    #st.write(f"Pixel size: {pixel_size}")
                    
                    # Create a mask for pixels with grade in the specified range
                    grade_mask = np.zeros_like(rst, dtype=bool)
                    
                    # For each pixel, calculate the maximum grade to any of its 8 neighbors
                    for i in range(1, rst.shape[0] - 1):
                        for j in range(1, rst.shape[1] - 1):
                            center_elev = rst[i, j]
                            
                            # Skip if elevation is above 2150m or is NaN
                            if np.isnan(center_elev) or center_elev > 2150:
                                continue
                            
                            # 8 neighbor offsets: (row, col)
                            neighbors = [
                                (-1, -1), (-1, 0), (-1, 1),
                                (0, -1),           (0, 1),
                                (1, -1),  (1, 0),  (1, 1)
                            ]
                            
                            # Check grade to each neighbor
                            for di, dj in neighbors:
                                ni, nj = i + di, j + dj
                                neighbor_elev = rst[ni, nj]
                                
                                # Skip if either pixel is NaN
                                if np.isnan(center_elev) or np.isnan(neighbor_elev):
                                    continue
                                
                                # Calculate elevation difference and distance
                                elev_diff = abs(neighbor_elev - center_elev)
                                distance = abs(pixel_size) * np.sqrt(di**2 + dj**2)
                                
                                # Calculate grade as percentage
                                grade_percent = (elev_diff / distance) * 100 if distance > 0 else 0
                                
                                # Mark pixel if grade is within the specified range
                                if grade_min <= grade_percent <= grade_max:
                                    grade_mask[i, j] = True
                                    break
                    
                    pixels_in_range = np.sum(grade_mask)
                    #st.write(f"Pixels with grade {grade_min}-{grade_max}% (below 2150m): {pixels_in_range}")
                    
                    window_transform = rasterio.windows.transform(window, src.transform)

                    # ============================================================================
                    # CREATE VISUALIZATIONS
                    # ============================================================================
                    
                    # Calculate extent for imshow
                    extent = [window_transform.c, 
                              window_transform.c + window_transform.a * rst.shape[1],
                              window_transform.f + window_transform.e * rst.shape[0], 
                              window_transform.f]

                    # Figure 3: Satellite
                    st.subheader("Satellite Imagery")
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    
                    ax3.set_xlim(extent[0], extent[1])
                    ax3.set_ylim(extent[2], extent[3])
                    
                    cxt.add_basemap(ax3,
                                    crs="EPSG:3979",
                                    source=cxt.providers.Esri.WorldImagery,
                                    attribution_size=6)
                    
                    ax3.set_title(f'Satellite Imagery')
                    ax3.set_xlabel('X Coordinate (EPSG:3979)')
                    ax3.set_ylabel('Y Coordinate (EPSG:3979)')
                    
                    st.pyplot(fig3)
                    
                    # Figure 1: Elevation Only
                    st.subheader("Elevation Only")
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    
                    im_elev2 = ax2.imshow(rst, cmap='terrain', extent=extent, origin='upper', 
                                         aspect='auto', interpolation='nearest', zorder=0)
                    ax2.set_title('Elevation Map (No Grade Overlay)')
                    ax2.set_xlabel('X Coordinate (EPSG:3979)')
                    ax2.set_ylabel('Y Coordinate (EPSG:3979)')
                    
                    plt.colorbar(im_elev2, ax=ax2, label='Elevation (meters)')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Figure 2: With Grade Overlay
                    st.subheader("Elevation Map with Grade")
                    fig1, ax1 = plt.subplots(figsize=(10, 8))
                    
                    im_elev1 = ax1.imshow(rst, cmap='terrain', extent=extent, origin='upper', 
                                         aspect='auto', interpolation='nearest', zorder=0)
                    ax1.set_title(f'Elevation with Grade Overlay ({grade_min}-{grade_max}%)')
                    ax1.set_xlabel('X Coordinate (EPSG:3979)')
                    ax1.set_ylabel('Y Coordinate (EPSG:3979)')
                    
                    # Add grade overlay
                    grade_overlay_masked = np.ma.masked_where(grade_mask, grade_mask.astype(float))
                    im_grade = ax1.imshow(grade_overlay_masked, cmap=plt.cm.Reds, alpha=0.99, 
                                         extent=extent, zorder=2, origin='upper', interpolation='nearest')
                    
                    plt.colorbar(im_elev1, ax=ax1, label='Elevation (meters)')
                    plt.tight_layout()
                    st.pyplot(fig1)

                    st.success("Grade Analysis complete!")
                    
                    
                    
        # ============================================================================
        # HYDRO ANALYSIS AND RESULTS TABLE
        # ============================================================================
                    
        @dataclass
        class Inputs:
            head_m: float
            flow_m3s: float
            population: int

        @dataclass
        class Assumptions: 
            rho: float = 1000.0
            g: float = 9.81             #the rest of these values could be moved to site specific input values or should at least have our research to back them up
            sys_eff: float = 0.75       #total system efficiency
            flow_fraction: float = 0.3  #assumed fraction of river flow that system is using
            hydro_cf: float = 0.7       #capacity factor - accounting for flow seasonality
            per_capita_kwh: float = 9900     
            diesel_l_per_kwh: float = 0.7
            diesel_fuel_price_per_l: float = 1.85 #$
            diesel_om_per_kwh: float = 0.10 #$
            diesel_em_per_kwh: float = 0.22 #$
            hydro_capex_per_kw: float = 2500 #$
            hydro_life_years: int = 30
            discount_rate: float = 0.07
            #hydro_lcoe_per_kwh: float = 0.12 #$
            #head_loss_fraction: float = 0.10

        @dataclass
        class Results: 
            turbine_type: str
            hydro_capacity_kw: float
            hydro_annual_kwh: float
            demand_annual_kwh: float 
            diesel_cost_per_kwh: float
            hydro_cost_per_kwh: float
            diesel_annual_cost: float
            hydro_annual_cost: float
            hydro_initial_cost: float
            annual_savings: float
            savings_percent: float
            suitability: str
            notes: list

        def select_turbine(head_m: float) -> str:
            H = head_m

            if H < 5:
                return "Reaction Turbines: Propeller; Kaplan"
            
            if 5 <= H < 20:
                return "Impulse Turbines: Cross-flow; Multi-jet Turgo \n         Reaction Turbines: Propeller; Kaplan"
            
            if 20 <= H < 100:
                return "Impulse Turbines: Cross-flow; Turgo; Multi-jet Pelton \n         Reaction Turbines: Francis; Pump-as-turbine"
            
            if H > 100:
                return "Impulse Trubines: Pelton; Turgo"


        def lcoe_calculation(capex_CAD: float, opex_annual_CADperYr: float, energy_annual_kwh: float, lifetime_yr: float, discount_rate_per: float, fuel_annual):
            
                #Simple LCOE Calculator
                #capex          : upfront capital cost ($)
                #opex_annual    : annual operating cost ($/year)
                #fuel_annual    : annual fuel cost ($/year)
                #energy_annual  : annual electricity generation (kWh/year)
                #lifetime       : project lifetime (years)
                #discount_rate  : discount rate (decimal, e.g., 0.06 for 6%)
            
                discounted_costs = 0
                discounted_energy = 0
                
                for t in range(1, lifetime_yr + 1):
                    discounted_costs += (opex_annual_CADperYr + fuel_annual) / (1 + discount_rate_per)**t
                    discounted_energy += (energy_annual_kwh) / (1 + discount_rate_per)**t

                # CapEx occurs at year 0 (not discounted over a year)
                discounted_costs += capex_CAD
                
                lcoe = discounted_costs / discounted_energy
                return lcoe

        def assess_micro_hydro(inputs: Inputs, assumptions: Assumptions = Assumptions()) -> Results:
            notes = []

            #Effective Flow
            effective_flow = inputs.flow_m3s * assumptions.flow_fraction 

            #Hydro capacity (kW)
            P_watts = assumptions.rho * assumptions.g * effective_flow * inputs.head_m * assumptions.sys_eff
            P_kw = P_watts/1000.0

            #Annual hydro energy (kwh)
            hydro_annual_kwh = P_kw * assumptions.hydro_cf * 8760 #hours/yr

            #Demand Estimate (kwh)
            demand_annual_kwh = inputs.population * assumptions.per_capita_kwh
        
            #Hydro LCOE 
            hydro_initial_cost = P_kw * 2500
            hydro_annual_opex = hydro_initial_cost * 0.06
            hydro_lcoe_per_kwh = lcoe_calculation(hydro_initial_cost, 
                                                hydro_annual_opex, 
                                                hydro_annual_kwh, 
                                                assumptions.hydro_life_years, 
                                                assumptions.discount_rate, 0)
            
            #Diesel LCOE 
            diesel_annual_opex = (assumptions.diesel_om_per_kwh + assumptions.diesel_em_per_kwh) * demand_annual_kwh
            diesel_annual_fuel_cost = assumptions.diesel_l_per_kwh * assumptions.diesel_fuel_price_per_l * demand_annual_kwh
            diesel_lcoe_per_kwh = lcoe_calculation(0, 
                                                diesel_annual_opex, 
                                                demand_annual_kwh, 
                                                assumptions.hydro_life_years, 
                                                assumptions.discount_rate, 
                                                diesel_annual_fuel_cost)

            #Annual Costs 
            diesel_annual_cost = diesel_lcoe_per_kwh * demand_annual_kwh

            displaced_kwh = min(hydro_annual_kwh, demand_annual_kwh)
            hydro_annual_cost = (displaced_kwh * hydro_lcoe_per_kwh) + (max(demand_annual_kwh - hydro_annual_kwh, 0) * diesel_lcoe_per_kwh)

            #Savings 
            annual_savings = diesel_annual_cost - hydro_annual_cost
            savings_percent = (annual_savings / diesel_annual_cost * 100) if diesel_annual_cost > 0 else 0.0

            #Turbine selection 
            turbine = select_turbine(inputs.head_m)

            #Suitability decision
            coverage_pct = (hydro_annual_kwh / demand_annual_kwh * 100) if demand_annual_kwh > 0 else 0.0
            capacity_ok = P_kw >= 50.0
            coverage_ok = coverage_pct >= 50.0
            savings_ok = savings_percent >= 20

            if inputs.head_m < 2: 
                suitability = "Not suitable (head to low)"
                notes.append("Head below ~2 m is typically impractical without major civil works.")
            elif inputs.flow_m3s <= 0: 
                suitability = "Not suitable (no flow)"
                notes.append("Flow must be > 0 m^3/s.")
            else: 
                if (coverage_ok or capacity_ok) and savings_ok:
                    suitability = "Good Fit!"
                elif (coverage_ok or capacity_ok) and not savings_ok:
                    suitability = "Technically viable, economic benifits are marginal"
                    notes.append("Consider grants, carbon pricing, or higher diesel cost; economics may improve.")
                elif savings_ok and not (coverage_ok or capacity_ok):
                    suitability = "Economically beneficial as supplement"
                    notes.append("Hydro can displace part-time; consider hybrid control strategy.")
                else: 
                    suitability = "Marginal; consider optimization"
                    notes.append("Explore intake optimization, higher efficiency, or battery/storage to raise capacity factor.")

            if assumptions.hydro_cf < 0.4: 
                notes.append("Low hydro capacify factor suggests strong seasonality; hybrid with diesel will likley be required.")
            if coverage_pct > 90: 
                notes.append("Hydro may exceed demands seasonally; consider spill or productive uses (e.g. heat, battery charging).")
            
            return Results(
                turbine_type=turbine,
                hydro_capacity_kw=round(P_kw, 2),
                hydro_annual_kwh=round(hydro_annual_kwh, 0),
                demand_annual_kwh=round(demand_annual_kwh, 0),
                diesel_cost_per_kwh=round(diesel_lcoe_per_kwh, 2),
                hydro_cost_per_kwh=round(hydro_lcoe_per_kwh, 2),
                diesel_annual_cost=round(diesel_annual_cost, 0),
                hydro_annual_cost=round(hydro_annual_cost, 0),
                hydro_initial_cost=round(hydro_initial_cost, 0),
                annual_savings=round(annual_savings, 0),
                savings_percent=round(savings_percent, 1),
                suitability=suitability,
                notes=notes
            )

        try:
            head = float(head) #NV: 50
            flow = float(flow_rate) #NV: 0.087
            pop = int(community_population) #NV: 197

            inputs = Inputs(head_m=head, flow_m3s=flow, population=pop)
            res = assess_micro_hydro(inputs)
            
            # Display results in a summary table
            st.subheader("3. Micro-Hydro Analysis Summary")
            
            # Create results dictionary for table display
            results_data = {
                "Metric": [
                    "Suitability",
                    "Turbine Type",
                    "Microhydro Capacity (kW)",
                    "Microhydro Annual Generation (kWh)",
                    "Community Demand Annual (kWh)",
                    "Diesel LCOE ($/kWh)",
                    "Microhydro LCOE ($/kWh)",
                    "Microhydro Capital Cost ($)",
                    "Diesel Annual Cost ($)",
                    "Post-Microhydro Annual Cost ($)",
                    "Annual Savings ($)",
                    "Savings Percentage (%)"
                ],
                "Value": [
                    res.suitability,
                    res.turbine_type,
                    res.hydro_capacity_kw,
                    res.hydro_annual_kwh,
                    res.demand_annual_kwh,
                    res.diesel_cost_per_kwh,
                    res.hydro_cost_per_kwh,
                    res.hydro_initial_cost,
                    res.diesel_annual_cost,
                    res.hydro_annual_cost,
                    res.annual_savings,
                    res.savings_percent
                ]
            }
            
            st.dataframe(results_data, use_container_width=True)
            
            # Display notes if any
            if res.notes:
                st.subheader("Notes")
                for note in res.notes:
                    st.write(f"• {note}")
            
        except ValueError: 
            st.error("Invalid input. Please enter numeric values for head, flow, and population.")
                   
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your inputs and try again.")

else:
    st.info("Set your parameters in the sidebar and click 'Run Analysis' to begin.")
