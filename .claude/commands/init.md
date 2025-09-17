# Project Initialization Command

## Command: init

### Purpose
Initialize understanding of the pyBlockGrid project by reading ONLY the summary documentation files.

### Instructions for Claude

When this command is invoked, you should:

1. **Read the following files in order:**
   - `context/summaries/1_codebase_structure.md` - Understand the project organization
   - `context/summaries/2_volkov_solver.md` - Learn the core algorithm implementation
   - `context/summaries/3_visualizations.md` - Understand visualization capabilities
   - `context/summaries/4_tests.md` - Learn about the test suite

2. **Do NOT read any other files** - The summaries contain all necessary information for initial familiarization

3. **After reading**, briefly acknowledge that you have:
   - Understood the project structure
   - Learned about the Volkov method implementation
   - Familiarized yourself with the visualization and testing components

4. **Then wait** for further instructions from the user

### Expected Response Format
```
I've familiarized myself with the pyBlockGrid project:
- ✓ Project structure and organization understood
- ✓ Volkov solver algorithm and implementation reviewed  
- ✓ Visualization capabilities noted
- ✓ Test suite structure understood

Ready for your instructions.
```

### Usage
This command should be run at the start of any new session to quickly bring the agent up to speed with the project without overwhelming it with implementation details.