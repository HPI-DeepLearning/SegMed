import React, { Component } from 'react'
import BrainView from './BrainView'
import { startedUpload, finishedUpload, receivedResults } from '../actions'
import { PropTypes } from 'react'
import { connect } from 'react-redux'

class AnalyzingPage extends Component {
  render() {
    return (
      <div className='analyze'>
        <div className='header'>
          <h1 className='headLine'>Tumor Viz</h1>
        </div>
        <div className='content'>
          <div className='leftContent'>
            <div className='picture'>
              <BrainView />
            </div>
            <div className='descriptionContainer'>
              <p>{this.props.regionTitle}</p>
            </div>
          </div>
          <div className='rightContent'>
            <div className='info'>
              <p>Type: {this.props.tumorType}</p>
              <p>Survival rate: {this.props.survivalRate}</p>
            </div>
            <div className="add">
              <label htmlFor="file-input">
                <a className='iconSmall'>
                  <i className="fa fa-plus" aria-hidden="true"></i>
                  <p>Upload a brain</p>
                </a>
              </label>
              <input id="file-input"
                className="fileInput"
                type="file"
                onChange={(e)=>this._handleImageChange(e)}/>
            </div>
          </div>
        </div>
      </div>
    )
  }

  uploadImage(url) {
    this.props.startedUpload()
    fetch('http://localhost:5000/tumor/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url,
      })
    }).then(response => response.json())
    .then((data) => {
      this.props.finishedUpload()
      this.props.receivedResults(data)
    }).catch((err) => {
      this.props.finishedUpload()
      console.error('ERROR!', err)
    })
  }

  _handleImageChange(e) {
    e.preventDefault()
    let reader = new FileReader()
    let file = e.target.files[0]
    reader.onloadend = () => this.uploadImage(reader.result)
    reader.readAsDataURL(file)
  }
}


AnalyzingPage.propTypes = {
  startedUpload: PropTypes.func.isRequired,
  receivedResults: PropTypes.func.isRequired,
  imageUrl: PropTypes.string,
  regionTitle: PropTypes.string,
  tumorType: PropTypes.string,
  survivalRate: PropTypes.string,
}


function mapStateToProps(state) {
  return {
    imageUrl: state.images.url,
    regionTitle: state.tumor.regionTitle,
    tumorType: state.tumor.tumorType,
    survivalRate: state.tumor.survivalRate,
  }
}

function mapDispatchToProps(dispatch) {
  return {
    startedUpload: () => {
      dispatch(startedUpload())
    },
    finishedUpload: () => {
      dispatch(finishedUpload())
    },
    receivedResults: (results) => {
      dispatch(receivedResults(results))
    },
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(AnalyzingPage)
