# Contributing to Credit Risk Prediction Pipeline

## 📋 Guidelines

### Code Quality
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to all functions
- Include type hints where possible
- Keep functions focused and modular

### Documentation
- Update README.md for major changes
- Add inline comments for complex logic
- Document new features in docs/
- Update this file if adding new guidelines

### Testing
- Run full pipeline before committing
- Verify no data leakage
- Check model performance metrics
- Validate data quality checks pass

### Commit Messages
- Use clear, descriptive messages
- Reference issue numbers if applicable
- Keep commits focused and atomic

### Pull Requests
- Provide clear description of changes
- Reference related issues
- Include before/after metrics if applicable
- Ensure all tests pass

## 🔍 Code Review Checklist

- [ ] No data leakage introduced
- [ ] Preprocessing order is correct
- [ ] All transformers fit on train only
- [ ] Validation strategy is sound
- [ ] Class imbalance is handled
- [ ] Code follows PEP 8
- [ ] Documentation is updated
- [ ] Tests pass successfully

## 📝 Documentation Standards

### Docstrings
```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

### Comments
```python
# Use comments for WHY, not WHAT
# Good: Calculate guarantee ratio to capture SBA coverage
guarantee_ratio = sba_guarantee / gross_approval

# Bad: Divide sba_guarantee by gross_approval
```

## 🚀 Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit with clear messages: `git commit -m "Add feature description"`
4. Push to branch: `git push origin feature/your-feature`
5. Create pull request with description
6. Address review comments
7. Merge when approved

## 🧪 Testing

### Before Committing
```bash
# Run the full pipeline
jupyter notebook notebooks/credit_risk_pipeline.ipynb

# Verify:
# - No errors during execution
# - Data quality validation passes
# - Model performance is as expected
# - No data leakage detected
```

### Key Checks
- [ ] Train/val/test split is stratified
- [ ] Imputers fit on train only
- [ ] Encoders fit on train only
- [ ] Scaler fit on train only
- [ ] No NaN values in final datasets
- [ ] No Inf values in numeric features
- [ ] Feature counts consistent across sets
- [ ] Target distribution maintained

## 📚 Resources

- **README.md** - Project overview
- **docs/TECHNICAL_FEATURE_ANALYSIS.md** - Technical details
- **docs/PROFESSIONAL_ML_REVIEW.md** - ML engineering review
- **notebooks/credit_risk_pipeline.ipynb** - Main pipeline

## ❓ Questions?

Refer to the comprehensive documentation in the `docs/` directory or review the notebook comments for implementation details.

---

**Last Updated:** March 16, 2026
