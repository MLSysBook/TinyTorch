# Marimo and NBGrader Compatibility

## Short Answer: ✅ No, Marimo badges won't break NBGrader

**Why:**
- Marimo badges are just **frontend UI elements** (JavaScript links)
- They don't modify notebook files
- NBGrader reads from actual `.ipynb` files, not from the website
- Badges just create links to open notebooks in Marimo's cloud service

## How It Works

### Marimo Badges (What We Added)
- **What they do**: Add a "🍃 Open in Marimo" link to notebook pages
- **What they don't do**: Modify notebook files or NBGrader metadata
- **Impact on NBGrader**: **None** - they're just links

### NBGrader Workflow
1. Instructors generate notebooks: `tito nbgrader generate MODULE`
2. NBGrader adds metadata to `.ipynb` files (grade_id, points, etc.)
3. Students work in notebooks (Jupyter, Colab, or Marimo)
4. Students submit notebooks back
5. NBGrader reads metadata from submitted `.ipynb` files

## Potential Considerations

### If Students Use Marimo to Edit Notebooks

**Scenario 1: Students open `.ipynb` in Marimo**
- ✅ Marimo can import Jupyter notebooks
- ✅ NBGrader metadata preserved (it's in the `.ipynb` file)
- ✅ Students submit `.ipynb` files back
- ✅ **No problem** - NBGrader works normally

**Scenario 2: Students convert to Marimo `.py` format**
- ⚠️ Marimo stores notebooks as `.py` files (not `.ipynb`)
- ⚠️ NBGrader metadata is in `.ipynb` format
- ⚠️ Converting to `.py` might lose NBGrader metadata
- ✅ **Solution**: Students should submit `.ipynb` files, not `.py` files

## Best Practice for Students

**For NBGrader assignments:**
1. Students can use Marimo to **view and learn** from notebooks
2. For **submissions**, students should work in `.ipynb` format (Jupyter/Colab)
3. Or convert marimo `.py` back to `.ipynb` before submitting

**For non-graded exploration:**
- Students can freely use Marimo's `.py` format
- Great for learning and experimentation
- No NBGrader concerns

## Recommendation

**Keep Marimo badges** - they're safe:
- ✅ Don't interfere with NBGrader
- ✅ Give students more options for learning
- ✅ Students can use Marimo for exploration
- ✅ For graded work, students use standard `.ipynb` workflow

**Add to student instructions:**
- "Marimo badges are for exploration and learning"
- "For NBGrader assignments, submit `.ipynb` files (not `.py` files)"
- "Marimo can import `.ipynb` files and preserve NBGrader metadata"

## Technical Details

### NBGrader Metadata Format
NBGrader stores metadata in notebook cell metadata:
```json
{
  "nbgrader": {
    "grade": true,
    "grade_id": "tensor_memory",
    "points": 2,
    "schema_version": 3
  }
}
```

### Marimo Format
Marimo stores notebooks as pure Python:
```python
# Cell 1
import numpy as np

# Cell 2  
def memory_footprint(self):
    return self.data.nbytes
```

**Conversion between formats:**
- `.ipynb` → `.py`: Possible, but NBGrader metadata might be lost
- `.py` → `.ipynb`: Possible, but NBGrader metadata won't be restored

## Conclusion

✅ **Marimo badges are safe** - they don't break NBGrader
✅ **Students can use Marimo** for learning and exploration
✅ **For graded work**, students should use `.ipynb` format
✅ **No changes needed** to NBGrader workflow

The badges are just convenient links - they don't interfere with the actual grading system!

