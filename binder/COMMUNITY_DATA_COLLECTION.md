# Community Data Collection Design

## Data We Collect (Privacy-Respecting)

### Required Fields
- **Country**: Geographic location (country-level only)
- **Setup Verified**: Confirmation that setup works

### Optional Fields (User Can Skip)
- **School/Institution**: University, bootcamp, or organization name
- **Course Type**: How they're using TinyTorch
  - Self-paced learning
  - University course
  - Bootcamp/training
  - Research project
  - Industry training
- **System Type**: Hardware/platform
  - Apple Silicon
  - Linux x86
  - Windows
  - Cloud (Colab/Binder)
- **Experience Level**: (Optional)
  - Beginner
  - Intermediate
  - Advanced

### What We DON'T Collect
- ❌ Personal name
- ❌ Email address
- ❌ Exact location (city/coordinates)
- ❌ IP address
- ❌ Any personally identifiable information

## Data Structure

### Submission JSON

```json
{
  "anonymous_id": "abc123...",  // Generated hash
  "timestamp": "2024-11-20T10:30:00Z",
  
  "location": {
    "country": "United States"  // Required
  },
  
  "institution": {
    "name": "Harvard University",  // Optional
    "type": "university"  // Optional: university, bootcamp, company, self-paced
  },
  
  "context": {
    "course_type": "university_course",  // Optional
    "experience_level": "intermediate"  // Optional
  },
  
  "system": {
    "type": "Apple Silicon",  // Optional
    "platform": "darwin",
    "python_version": "3.11.0"
  },
  
  "progress": {
    "setup_verified": true,
    "milestones_passed": 0,  // Will update later
    "modules_completed": 0   // Will update later
  }
}
```

## Collection Flow

### Interactive Prompt

```bash
tito community join

🌍 Join the TinyTorch Global Community

This will add your location to the public community map.
All information is optional and completely anonymized.

📍 Country: [Auto-detected: United States] 
   (Press Enter to use detected, or type different country)

🏫 School/Institution (optional):
   Examples: "Harvard University", "Stanford", "Self-paced"
   [Press Enter to skip]

📚 Course Type (optional):
   [1] Self-paced learning
   [2] University course
   [3] Bootcamp/training
   [4] Research project
   [5] Industry training
   [6] Skip
   Choose [1-6]: 

💻 System Type (optional):
   [Auto-detected: Apple Silicon]
   [Press Enter to use detected, or type different]

🎓 Experience Level (optional):
   [1] Beginner
   [2] Intermediate
   [3] Advanced
   [4] Skip
   Choose [1-4]: 

📊 What will be shared:
   • Country: United States ✅
   • Institution: Harvard University ✅
   • Course Type: University course ✅
   • System Type: Apple Silicon ✅
   • No personal information ✅

🔒 Privacy: Completely anonymized, country-level location only

Continue? [y/N]: y

✅ You've joined the TinyTorch Community!

📍 Location: United States
🏫 Institution: Harvard University
🌍 View map: https://tinytorch.ai/community

🎖️ You're builder #1,234 on the global map!

💡 Your institution will appear on the map (if provided)
```

## Map Visualization Features

### What the Map Shows

**Country View:**
- Dots/countries with builder counts
- "1,234 builders in 45 countries"

**Institution View** (Optional Filter):
- "Builders from 234 institutions"
- Top institutions by builder count
- "Harvard University: 15 builders"
- "Stanford: 12 builders"
- "Self-paced: 456 builders"

**Course Type Breakdown:**
- "University courses: 234"
- "Self-paced: 456"
- "Bootcamps: 89"
- "Research: 123"

**Diversity Stats:**
- "Builders from 45 countries"
- "234 institutions represented"
- "5 course types"
- "Diverse experience levels"

## Privacy Considerations

### Institution Privacy

**Options:**
1. **Show institution names** (if provided)
   - Pros: More engaging, shows diversity
   - Cons: Might identify users in small programs

2. **Show institution counts only**
   - Pros: More private
   - Cons: Less engaging

3. **Hybrid approach** (Recommended)
   - Show institution names if ≥3 builders from that institution
   - Otherwise: "Other institutions: 5 builders"
   - Protects privacy while showing diversity

### Consent Flow

**Clear messaging:**
```
⚠️  Institution Information

If you provide your school/institution name, it may appear on the public map.

🔒 Privacy Protection:
   • Institution names only shown if ≥3 builders from that institution
   • No personal names or identifiers
   • Completely anonymized

Provide institution? [y/N]: 
```

## Map Features

### Interactive Map

**Country Level:**
- Click country → See stats:
  - "United States: 456 builders"
  - "Top institutions: Harvard (15), Stanford (12), MIT (10)"
  - "Course types: University (234), Self-paced (189)"

**Institution Filter:**
- Filter by institution type
- Show: Universities, Bootcamps, Self-paced, etc.
- See geographic distribution

**Course Type View:**
- Color-code by course type
- Show: "Where are university students?"
- Show: "Where are self-paced learners?"

### Stats Dashboard

```
🌍 TinyTorch Community

📊 Global Stats:
   • 1,234 builders worldwide
   • 45 countries
   • 234 institutions
   • 5 course types

🏫 Top Institutions:
   1. Harvard University: 15 builders
   2. Stanford: 12 builders
   3. MIT: 10 builders
   4. Self-paced: 456 builders
   ...

🌎 Geographic Diversity:
   • United States: 456 builders
   • India: 234 builders
   • United Kingdom: 123 builders
   ...

📚 Course Types:
   • Self-paced: 456 (37%)
   • University: 234 (19%)
   • Bootcamp: 89 (7%)
   ...
```

## Benefits of Collecting This Data

### For Community
- **Visual diversity**: See global reach
- **Institutional connections**: "Wow, people from my school!"
- **Course type insights**: Understand how TinyTorch is used
- **Motivation**: "There are builders from 234 institutions!"

### For Users
- **Representation**: "I'm representing my school!"
- **Connection**: Find others from same institution
- **Pride**: "My institution is on the map!"

### For Project
- **Adoption tracking**: See where TinyTorch is used
- **Diversity metrics**: Geographic and institutional diversity
- **Success stories**: "Used in 234 institutions worldwide"

## Implementation

### Data Collection

**Command**: `tito community join`

**Flow:**
1. Auto-detect country (using system locale or geolocation API)
2. Ask for institution (optional)
3. Ask for course type (optional)
4. Auto-detect system type
5. Ask for experience level (optional)
6. Show summary
7. Get consent
8. Generate submission

### Privacy Protection

**Institution Anonymization:**
- If <3 builders from institution → Show as "Other institutions"
- If ≥3 builders → Show institution name
- Protects privacy while showing diversity

**Data Storage:**
- Anonymous ID (hash, not personal)
- No personal identifiers
- Country-level only (not city)
- Optional fields can be skipped

## Recommended Fields

### Required
- ✅ Country

### Highly Recommended (Optional)
- ✅ Institution/School name
- ✅ Course type

### Nice to Have (Optional)
- System type (auto-detected)
- Experience level
- Milestone progress (updates later)

### Skip
- Personal name
- Email
- Exact location
- Any PII

## Example Map Entry

**What users see:**
```
📍 United States
   • 456 builders
   • Top institutions: Harvard (15), Stanford (12), MIT (10)
   • Course types: University (234), Self-paced (189)
```

**What gets stored:**
```json
{
  "country": "United States",
  "institution": "Harvard University",
  "course_type": "university_course",
  "anonymous_id": "abc123..."
}
```

This creates a rich, engaging community map while respecting privacy! 🌍✨

