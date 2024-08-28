import { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types'
import { nanoid } from 'nanoid';

import '../css/UsePhoto.css'

function UsePhoto(props) {
    const setUsePhoto = props.setUsePhoto;

    const [file, setFile] = useState();
    const [imgLoaded, setImgLoaded] = useState(false);
    const [recText, setRecText] = useState();
    const [recs, setRecs] = useState();
    const [loadingRecs, setLoadingRecs] = useState(false);

    const imgRef = useRef(null);
    const typewriteIntervalId = useRef(null);

    const startOver = () => {
        setRecs(null);
        setRecText(null);
        setDisplayRecs(null);
        setImgLoaded(false);
        clearInterval(typewriteIntervalId.current);
    }

    useEffect(() => { // Setting img source to user-uploaded file
        if (file) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            setImgLoaded(true);
            
            reader.addEventListener("load", () => {
                imgRef.current.src = reader.result;
            });
        }
    }, [file])

    const [displayRecs, setDisplayRecs] = useState();
    useEffect(() => { // Parsing recText and setting recs
        if (recText) {
            setLoadingRecs(false);
    
            const bulletPoints = recText.split('- **').splice(1);

            setRecs(bulletPoints.map((b) => {
                let pointSplit = b.split("**: ");
                return [pointSplit[0], pointSplit[1].trim()];                    
            }));

            setDisplayRecs(bulletPoints.map(() => ["", ""]));                
        }
    }, [recText]);

    const typewriteEffect = (intervalId) => { // Creates typewriter effect for recommendation text using setInterval
        let i = 0; // Represents index in recs
        let j = 0; // Represents index in list at recs[i]
        let c = 0; // Represents index in string at recs[i][j]

        intervalId.current = setInterval(() => {
            if (i < recs.length) {
                if (j < recs[0].length) {
                    if (c < recs[i][j].length) {
                        // Added character at recs[i][j][c] to displayRecs. displayRecs is then re-rendered on screen.
                        setDisplayRecs(displayRecs.map((r) => {
                            if (r[j] === displayRecs[i][j]) {
                                r[j] += recs[i][j][c];
                            }
        
                            return r;
                        }));

                        c++;

                    } else {
                        j++;
                        c = 0;
                    }   

                } else {
                    i++;
                    j = 0;
                }
                
            } else {
                return;
            }

        }, 8);
    };

    useEffect(() => { // Once recs are gathered and parsed, start displaying them using typewriter effect
        if (displayRecs && displayRecs[0][0] === '') {
            typewriteEffect(typewriteIntervalId);
            
            if (typewriteIntervalId) return () => clearInterval(typewriteIntervalId);            
        }

    }, [displayRecs])

    const handleSubmit = async () => { // Sends uploaded file in POST request to back end. Then handles outfit recs from back end.
        setLoadingRecs(true);
        const formData = new FormData();
        formData.append("file", file);

        const options = {
            method: "POST",
            body: formData
        }

        const response = await (await fetch("http://localhost:8080/upload-photo/", options)).json();
        setRecText(response.text);
    }

    return(
        <div className="use-photo">
            {!loadingRecs && !recs &&
                <div className='img-select'>
                    {imgLoaded && <img ref={imgRef} className='user-img' />} 
                    <label htmlFor='img-input' className='img-input-label'>Choose an Image</label>   
                    <input id='img-input' onChange={(e) => setFile(e.target.files[0])} name='image' type='file' accept='.jpg, .png, .jpeg' />
                    {imgLoaded && <button className='btn' id="use-photo-btn" onClick={handleSubmit}>Submit Image</button>}
                    <button className='btn' id="use-photo-btn" onClick={() => setUsePhoto(false)}>Home</button>
                </div>
            }

            {loadingRecs &&
                <div className='process-status'>
                    <div className="loading" />
                    <p>Getting Recommendations</p>
                </div>    
            } 

            {recs && 
                <h2 className='recs-header'>Outfit Recommendations:</h2>
            }

            {displayRecs && 
                <div className="recs-box">
                    <ul className="recs-list">
                        {displayRecs.map((r) => 
                            <li key={nanoid()}>
                                <h3>{r[0]}</h3>
                                <p>{r[1]}</p>
                            </li>                            
                        )}
                    </ul>
                </div>
            }

            {recs && <button className='btn' id="use-photo-btn" onClick={startOver}>Choose Another Photo</button>}         
            {recs && <button className='btn' id="use-photo-btn" onClick={() => setUsePhoto(false)}>Home</button>}
        </div>
    );
}


UsePhoto.propTypes = {
    setUsePhoto: PropTypes.func
}


export default UsePhoto;