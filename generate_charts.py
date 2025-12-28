import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional business charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('tikili_listings.csv')

# Convert posted_date to datetime
df['posted_date'] = pd.to_datetime(df['posted_date'])

# Create output directory
import os
os.makedirs('charts', exist_ok=True)

print("Generating business insight charts...")

# ============================================================================
# CHART 1: Average Price by District (Key Market Intelligence)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
district_avg = df.groupby('district')['price_usd'].mean().sort_values(ascending=True)
colors = ['#e74c3c' if x > district_avg.mean() else '#3498db' for x in district_avg]
district_avg.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('District', fontsize=12, fontweight='bold')
ax.set_title('Average Property Prices by District in Baku', fontsize=14, fontweight='bold', pad=20)
ax.axvline(district_avg.mean(), color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Market Average: ${district_avg.mean():,.0f}')
ax.legend()
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(district_avg):
    ax.text(v + 2000, i, f'${v:,.0f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/01_avg_price_by_district.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1: Average Price by District")

# ============================================================================
# CHART 2: Price Per Square Meter by District (Investment Value Analysis)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
district_psm = df.groupby('district')['price_per_sqm_usd'].mean().sort_values(ascending=True)
colors = ['#e74c3c' if x > district_psm.mean() else '#27ae60' for x in district_psm]
district_psm.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Price per Square Meter (USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('District', fontsize=12, fontweight='bold')
ax.set_title('Price Per Square Meter by District - Investment Density', fontsize=14, fontweight='bold', pad=20)
ax.axvline(district_psm.mean(), color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Market Average: ${district_psm.mean():,.0f}/sqm')
ax.legend()
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(district_psm):
    ax.text(v + 10, i, f'${v:,.0f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/02_price_per_sqm_by_district.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2: Price Per Square Meter by District")

# ============================================================================
# CHART 3: Market Inventory by Property Type (Supply Analysis)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
property_counts = df['property_type'].value_counts()
colors_prop = ['#3498db', '#e74c3c', '#f39c12']
bars = ax.bar(property_counts.index, property_counts.values, color=colors_prop, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Listings', fontsize=12, fontweight='bold')
ax.set_xlabel('Property Type', fontsize=12, fontweight='bold')
ax.set_title('Market Inventory Distribution by Property Type', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('charts/03_inventory_by_property_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3: Market Inventory by Property Type")

# ============================================================================
# CHART 4: Average Price by Property Type (Product Segmentation)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
type_avg = df.groupby('property_type')['price_usd'].mean().sort_values(ascending=False)
bars = ax.bar(type_avg.index, type_avg.values, color=['#e74c3c', '#3498db', '#27ae60'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_xlabel('Property Type', fontsize=12, fontweight='bold')
ax.set_title('Average Selling Price by Property Type', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('charts/04_avg_price_by_property_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4: Average Price by Property Type")

# ============================================================================
# CHART 5: Listing Volume Over Time (Market Activity Trends)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
df['month'] = df['posted_date'].dt.to_period('M')
monthly_counts = df.groupby('month').size()
monthly_counts.index = monthly_counts.index.to_timestamp()
ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=3, markersize=8, color='#3498db')
ax.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.3, color='#3498db')
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of New Listings', fontsize=12, fontweight='bold')
ax.set_title('Monthly Listing Activity - Market Supply Trends', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)
p = np.poly1d(z)
ax.plot(monthly_counts.index, p(range(len(monthly_counts))), "--", color='red', linewidth=2, alpha=0.7, label='Trend')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/05_listing_volume_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5: Listing Volume Over Time")

# ============================================================================
# CHART 6: Room Distribution (Demand Profile)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
room_counts = df['rooms'].value_counts().sort_index()
bars = ax.bar(room_counts.index, room_counts.values, color='#9b59b6', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Rooms', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Listings', fontsize=12, fontweight='bold')
ax.set_title('Property Distribution by Room Count - Market Composition', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/06_room_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 6: Room Distribution")

# ============================================================================
# CHART 7: Price by Room Count (Size-Based Pricing)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
room_price = df.groupby('rooms')['price_usd'].mean().sort_index()
bars = ax.bar(room_price.index, room_price.values, color='#e67e22', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Rooms', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_title('Average Property Price by Room Count', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/07_price_by_room_count.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 7: Price by Room Count")

# ============================================================================
# CHART 8: Metro Accessibility Impact (Location Premium)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
df['has_metro'] = df['metro'].notna()
metro_comparison = df.groupby('has_metro')['price_usd'].mean()
metro_labels = ['No Metro Access', 'Near Metro Station']
colors_metro = ['#95a5a6', '#27ae60']
bars = ax.bar(metro_labels, metro_comparison.values, color=colors_metro, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_title('Metro Accessibility Premium - Impact on Property Values', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels and percentage difference
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
# Calculate and show premium
premium = ((metro_comparison.iloc[1] - metro_comparison.iloc[0]) / metro_comparison.iloc[0] * 100)
ax.text(0.5, max(metro_comparison.values) * 0.5,
        f'Metro Premium:\n+{premium:.1f}%',
        ha='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
plt.tight_layout()
plt.savefig('charts/08_metro_accessibility_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 8: Metro Accessibility Impact")

# ============================================================================
# CHART 9: Repair Condition Impact on Pricing (Quality Premium)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
repair_price = df.groupby('repair')['price_usd'].mean().sort_values(ascending=True)
colors_repair = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(repair_price)))
bars = repair_price.plot(kind='barh', ax=ax, color=colors_repair, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Repair Condition', fontsize=12, fontweight='bold')
ax.set_title('Impact of Property Condition on Market Value', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, v in enumerate(repair_price):
    ax.text(v + 3000, i, f'${v:,.0f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/09_repair_condition_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 9: Repair Condition Impact")

# ============================================================================
# CHART 10: Price Distribution Analysis (Market Segmentation)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
# Create price ranges
price_ranges = ['Under $50k', '$50k-$100k', '$100k-$150k', '$150k-$200k', '$200k-$300k', 'Over $300k']
df['price_range'] = pd.cut(df['price_usd'],
                            bins=[0, 50000, 100000, 150000, 200000, 300000, float('inf')],
                            labels=price_ranges)
range_counts = df['price_range'].value_counts().sort_index()
bars = ax.bar(range(len(range_counts)), range_counts.values, color='#1abc9c', edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(range_counts)))
ax.set_xticklabels(range_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Properties', fontsize=12, fontweight='bold')
ax.set_xlabel('Price Range', fontsize=12, fontweight='bold')
ax.set_title('Market Price Distribution - Buyer Segments', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/10_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 10: Price Distribution Analysis")

# ============================================================================
# CHART 11: Top Districts by Listing Volume (Supply Concentration)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
district_counts = df['district'].value_counts().sort_values(ascending=True)
colors_dist = ['#3498db' if x < district_counts.mean() else '#e74c3c' for x in district_counts]
district_counts.plot(kind='barh', ax=ax, color=colors_dist, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Listings', fontsize=12, fontweight='bold')
ax.set_ylabel('District', fontsize=12, fontweight='bold')
ax.set_title('Market Supply Concentration by District', fontsize=14, fontweight='bold', pad=20)
ax.axvline(district_counts.mean(), color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {district_counts.mean():.0f}')
ax.legend()
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, v in enumerate(district_counts):
    ax.text(v + 2, i, f'{int(v)} ({v/len(df)*100:.1f}%)', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/11_supply_concentration.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 11: Supply Concentration by District")

# ============================================================================
# CHART 12: Document Type Distribution (Legal Structure)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
doc_counts = df['document'].value_counts()
bars = ax.bar(doc_counts.index, doc_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Properties', fontsize=12, fontweight='bold')
ax.set_xlabel('Document Type', fontsize=12, fontweight='bold')
ax.set_title('Property Ownership Documentation Types', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('charts/12_document_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 12: Document Type Distribution")

# ============================================================================
# CHART 13: Property Type Distribution by District (Market Mix)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
district_type = pd.crosstab(df['district'], df['property_type'])
district_type_pct = district_type.div(district_type.sum(axis=1), axis=0) * 100
district_type_pct.plot(kind='barh', stacked=True, ax=ax,
                        color=['#3498db', '#e74c3c', '#f39c12'],
                        edgecolor='black', linewidth=0.5)
ax.set_xlabel('Percentage of Listings (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('District', fontsize=12, fontweight='bold')
ax.set_title('Property Type Mix by District - Market Composition Analysis', fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/13_property_mix_by_district.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 13: Property Mix by District")

# ============================================================================
# CHART 14: Average Area by Property Type (Size Comparison)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
area_by_type = df.groupby('property_type')['area'].mean().sort_values(ascending=False)
bars = ax.bar(area_by_type.index, area_by_type.values, color=['#9b59b6', '#1abc9c', '#f39c12'],
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Area (Square Meters)', fontsize=12, fontweight='bold')
ax.set_xlabel('Property Type', fontsize=12, fontweight='bold')
ax.set_title('Average Property Size by Type', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f} sqm',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('charts/14_avg_area_by_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 14: Average Area by Property Type")

# ============================================================================
# CHART 15: Price Trends Over Time by Property Type
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
df['month'] = df['posted_date'].dt.to_period('M')
monthly_price_by_type = df.groupby(['month', 'property_type'])['price_usd'].mean().unstack()
monthly_price_by_type.index = monthly_price_by_type.index.to_timestamp()

for col in monthly_price_by_type.columns:
    ax.plot(monthly_price_by_type.index, monthly_price_by_type[col],
            marker='o', linewidth=2.5, markersize=6, label=col)

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Price (USD)', fontsize=12, fontweight='bold')
ax.set_title('Price Trends by Property Type Over Time', fontsize=14, fontweight='bold', pad=20)
ax.legend(title='Property Type', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/15_price_trends_by_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 15: Price Trends by Property Type")

print("\n" + "="*60)
print("✓ All charts generated successfully in the 'charts/' directory")
print("="*60)
