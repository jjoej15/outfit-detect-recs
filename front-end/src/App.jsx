import { useState, useEffect } from 'react';
import UseCamera from './components/UseCamera';
import UsePhoto from './components/UsePhoto';
import githubLogo from './assets/github-mark.svg';
import './css/App.css';

// Favicon rgb val is (223, 52, 52)

function App() {
  const [useCamera, setUseCamera] = useState(false);
  const [usePhoto, setUsePhoto] = useState(false);

  return (
    <div className='app'>
      {useCamera && <UseCamera />}
      {usePhoto && <UsePhoto />}

      {!useCamera && !usePhoto &&
        <div className='home'>
          <h1 className='app-title'>FitDetect</h1>
          <h3 className='app-desc'>Outfit Detection/Recommendation Engine</h3>

          <p className='credits'>Project by Joe Anderson <a href="https://github.com/jjoej15" target="_blank"><img src={githubLogo} className='github-logo' alt='github logo' /></a>. 
          Source code located <a href="https://github.com/jjoej15/outfit-detect-recs" target="_blank">here</a>.</p>

          <p className='app-details'>
            Get recommendations for your outfit using your webcam or an uploaded photo.
          </p>

          <button className='btn' onClick={setUseCamera}>Use Camera</button>
          <button className='btn' onClick={setUsePhoto}>Use Photo</button>
        </div>
      }
    </div>

  );
}

export default App
