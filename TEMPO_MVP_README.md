# TEMPO Visualization MVP

This project provides a locally runnable web application demonstrating the TEMPO (Threshold-based Exploration with Multipath Parallel Output) generation process with dynamic visualization of parallel token sets and pruning using Mistral-7B.

## Architecture

The application consists of two main components:

1. **Backend (Python/FastAPI)**:
   - Implements the TEMPO core logic with parallel token generation
   - Provides an API endpoint for text generation with various parameters
   - Uses Mistral-7B with MPS acceleration
   - Implements dynamic hybrid pruning strategy

2. **Frontend (SvelteKit/D3.js)**:
   - Provides an interactive UI for controlling generation parameters
   - Visualizes the token generation process step by step
   - Shows both original parallel token sets and pruned tokens
   - Implemented with responsive design using Tailwind CSS

## Backend Features

- **Model Loading**: Loads Mistral-7B with the CustomParallelAttentionModel wrapper
- **TEMPO Core Logic**: Implements parallel token generation based on threshold
- **Pruning Implementation**: Includes the Dynamic Hybrid Pruning strategy
- **API Endpoint**: FastAPI endpoint that accepts prompt and generation parameters
- **Invariant Programming**: Uses explicit invariant checks throughout the code

## Frontend Features

- **Interactive Controls**: Adjust threshold, tokens, pruning parameters, etc.
- **Dynamic Visualization**: Step-by-step view of token generation with D3.js
- **Token Tables**: Detailed view of tokens at each generation step
- **Responsive Design**: Works on different screen sizes

## Running the Application

### 1. Start the Backend

First, make sure you have the required Python dependencies:

```bash
pip install -r requirements.txt
```

Then start the FastAPI server:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs.

### 2. Start the Frontend

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:5173.

## Usage

1. Open the frontend in your browser
2. Enter a prompt in the text area
3. Adjust generation parameters as needed:
   - **Threshold**: Controls the probability threshold for token selection
   - **Max Tokens**: Maximum number of tokens to generate
   - **Use Pruning**: Enable/disable token pruning
   - **Pruning Strategy**: Choose between coherence, diversity, or hybrid
   - **Diversity Steps**: Number of steps to use diversity pruning before switching to coherence
   - **Coherence Threshold**: Threshold for pruning tokens based on attention coherence
   - **Diversity Clusters**: Number of clusters for diversity-based pruning
4. Click "Generate Text" to start the generation process
5. View the generated text and explore the token sets at each generation step
6. Use the step navigation to move through the generation process

## Implementation Notes

- The backend uses a singleton pattern to keep the model loaded in memory
- The frontend uses D3.js for visualization with reactive updates
- The application follows invariant programming principles with explicit validation and error handling
- The visualization shows which tokens were pruned vs. kept with color coding (blue for kept, red for pruned) 