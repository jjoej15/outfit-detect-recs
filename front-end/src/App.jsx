import { useState } from 'react';

import UseCamera from './components/UseCamera';
import UsePhoto from './components/UsePhoto';
import githubLogo from './assets/github-mark.svg';

import './css/App.css';

function App() {
  const [useCamera, setUseCamera] = useState(false);
  const [usePhoto, setUsePhoto] = useState(false);

  return (
    <div className='app'>
      {useCamera && <UseCamera setUseCamera={setUseCamera} />}
      {usePhoto && <UsePhoto setUsePhoto={setUsePhoto} />}

      {/* Home page */}
      {!useCamera && !usePhoto &&
        <div className='home'>
          <h1 className='app-title'>FitDetect</h1>
          <h3 className='app-desc'>Outfit Detection/Recommendation Engine</h3>

          <p className='credits'>Project by Joe Anderson <a href="https://github.com/jjoej15" target="_blank"><img src={githubLogo} className='github-logo' alt='github logo' /></a>. 
          Source code located <a href="https://github.com/jjoej15/outfit-detect-recs" target="_blank">here</a>.</p>

          <p className='app-details'>
            Get recommendations for your outfit using your webcam or an uploaded photo.
            Leveraging deep learning, computer vision, and LLMs to improve your style.
          </p>

          <button className='btn' onClick={() => setUseCamera(true)}>Use Camera</button>
          <button className='btn' onClick={() => setUsePhoto(true)}>Use Photo</button>
        </div>
      }
    </div>

  );
}

export default App
