# Intelligent Missing Columns Display

PySuricata now features an intelligent missing columns display system that automatically adapts to different dataset sizes and provides an optimal user experience for exploring missing data patterns.

## Features

### 🎯 **Dynamic Display Limits**
The system automatically adjusts how many missing columns to show based on dataset size:

- **Small datasets** (≤10 columns): Show all columns with missing data
- **Medium datasets** (11-50 columns): Show up to 10 columns initially
- **Large datasets** (51-200 columns): Show up to 12 columns initially  
- **Huge datasets** (>200 columns): Show up to 15 columns initially

### 🔍 **Smart Filtering**
Only displays columns with meaningful missing data:
- **Default threshold**: 0.5% missing data
- **Configurable**: Set custom thresholds via configuration
- **Noise reduction**: Filters out columns with insignificant missing patterns

### 📱 **Expandable UI**
For datasets with many missing columns:
- **Clean initial view**: Shows most important missing columns first
- **Expand functionality**: "Show X more..." button reveals additional columns
- **Smooth interaction**: JavaScript-powered expand/collapse with smooth animations
- **Accessibility**: Proper ARIA labels and keyboard navigation

### 🎨 **Visual Design**
- **Consistent styling**: Matches the existing PySuricata design system
- **Severity indicators**: Color-coded bars (low/medium/high missing percentages)
- **Responsive layout**: Works well on different screen sizes
- **Hover tooltips**: Detailed information on hover

## Configuration

### Engine Configuration

```python
from pysuricata.config import EngineConfig

config = EngineConfig(
    # Missing columns display options
    missing_columns_threshold_pct=0.5,    # Minimum missing % to display
    missing_columns_max_initial=8,        # Max columns shown initially
    missing_columns_max_expanded=25,      # Max columns when expanded
)
```

### API Usage

```python
from pysuricata.report import build_report
from pysuricata.config import EngineConfig

# Configure missing columns display
config = EngineConfig(
    missing_columns_threshold_pct=1.0,  # Only show columns with >1% missing
    missing_columns_max_initial=10,     # Show up to 10 columns initially
    missing_columns_max_expanded=30,    # Show up to 30 when expanded
)

# Generate report with custom configuration
html = build_report(df, config=config)
```

## Algorithm Details

### Dynamic Limit Calculation

```python
def get_initial_display_limit(n_cols: int, n_rows: int) -> int:
    """Determine initial display limit based on dataset characteristics."""
    if n_cols <= 10:
        return min(n_cols, 8)      # Show all for small datasets
    elif n_cols <= 50:
        return min(10, 8)          # Medium datasets
    elif n_cols <= 200:
        return min(12, 8)          # Large datasets
    else:
        return min(15, 8)          # Huge datasets
```

### Threshold Filtering

```python
def filter_significant_missing(miss_list, threshold_pct=0.5):
    """Filter out columns with insignificant missing data."""
    return [
        item for item in miss_list 
        if item[1] >= threshold_pct  # item[1] is missing percentage
    ]
```

### Expandable UI Logic

The system automatically determines when to show the expand functionality:

```python
needs_expandable = len(significant_missing) > initial_limit
```

## Examples

### Small Dataset (5 columns)
```
┌─────────────────────────────────────┐
│ Top missing columns                 │
├─────────────────────────────────────┤
│ col1     ████████████████ 15.0%    │
│ col2     ██████████ 8.0%           │
│ col3     ████████ 5.0%             │
│ col4     ██████ 3.0%               │
│ col5     ██ 1.0%                   │
└─────────────────────────────────────┘
```

### Large Dataset (100 columns)
```
┌─────────────────────────────────────┐
│ Top missing columns                 │
├─────────────────────────────────────┤
│ col1     ████████████████ 25.0%    │
│ col2     ██████████████ 20.0%      │
│ col3     ████████████ 15.0%        │
│ col4     ██████████ 12.0%          │
│ col5     ████████ 8.0%             │
│ col6     ██████ 6.0%               │
│ col7     ██████ 5.0%               │
│ col8     ████ 3.0%                 │
├─────────────────────────────────────┤
│ [Show 17 more...] ▼                │
└─────────────────────────────────────┘
```

### Huge Dataset (500 columns)
```
┌─────────────────────────────────────┐
│ Top missing columns                 │
├─────────────────────────────────────┤
│ col1     ████████████████ 30.0%    │
│ col2     ████████████████ 28.0%    │
│ col3     ███████████████ 25.0%     │
│ ... (12 more columns) ...           │
├─────────────────────────────────────┤
│ [Show 35 more...] ▼                │
└─────────────────────────────────────┘
```

## Benefits

### 🚀 **Performance**
- **Reduced DOM size**: Only renders visible columns initially
- **Faster rendering**: Less HTML to process and display
- **Smooth interactions**: JavaScript handles expand/collapse efficiently

### 👥 **User Experience**
- **Clean interface**: No overwhelming lists of columns
- **Progressive disclosure**: Users can explore more data as needed
- **Intelligent defaults**: Works well out of the box for most datasets
- **Customizable**: Power users can adjust thresholds and limits

### 🔧 **Developer Experience**
- **Modular design**: Clean separation of concerns
- **Comprehensive tests**: Full test coverage for all scenarios
- **Well documented**: Clear API and configuration options
- **Extensible**: Easy to add new features or modify behavior

## Technical Implementation

### Architecture

```
MissingColumnsAnalyzer
├── analyze_missing_columns()
├── _get_initial_display_limit()
└── _get_expanded_display_limit()

MissingColumnsRenderer  
├── render_missing_columns_html()
├── _render_columns_list()
└── _get_severity_class()

create_missing_columns_renderer()  # Factory function
```

### Files Modified

- `pysuricata/render/missing_columns.py` - New intelligent analysis system
- `pysuricata/render/html.py` - Updated to use new system
- `pysuricata/compute/manifest.py` - Updated manifest generation
- `pysuricata/static/css/style.css` - Added expandable UI styles
- `pysuricata/config.py` - Added configuration options
- `tests/test_missing_columns.py` - Comprehensive test suite

### Browser Compatibility

- **Modern browsers**: Full functionality with JavaScript
- **Fallback**: Graceful degradation for older browsers
- **Accessibility**: ARIA labels and keyboard navigation support

## Migration Guide

### From Previous Version

The new system is **fully backward compatible**. Existing reports will automatically benefit from:

- ✅ Dynamic display limits based on dataset size
- ✅ Smart filtering of insignificant missing data
- ✅ Expandable UI for large datasets
- ✅ Improved visual design

### Custom Configurations

If you were previously modifying the hardcoded limit of 5 columns, you can now use:

```python
# Old way (hardcoded)
# Limited to 5 columns always

# New way (configurable)
config = EngineConfig(
    missing_columns_threshold_pct=1.0,  # Custom threshold
    missing_columns_max_initial=10,     # Custom initial limit
    missing_columns_max_expanded=20,    # Custom expanded limit
)
```

## Future Enhancements

Potential future improvements:

- **Search/filter functionality**: Find specific columns by name
- **Export options**: Export missing data summary to CSV
- **Advanced analytics**: Trend analysis for missing data patterns
- **Interactive charts**: Click to drill down into missing data details
- **Custom thresholds per column type**: Different thresholds for numeric vs categorical

## Contributing

To contribute to the missing columns intelligence system:

1. **Tests**: Add test cases in `tests/test_missing_columns.py`
2. **Documentation**: Update this file for new features
3. **Configuration**: Add new options to `EngineConfig` if needed
4. **Styling**: Update CSS in `pysuricata/static/css/style.css`

The system is designed to be modular and extensible, making it easy to add new features while maintaining backward compatibility.
