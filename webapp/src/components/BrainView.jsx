import React, { Component } from 'react'
import { PropTypes } from 'react'
import ImageTracer from 'imagetracerjs'
import { selectRegion } from '../actions'
import { connect } from 'react-redux'

const IMG_SIZE = "256px"

class BrainView extends Component {

  componentDidMount() {
    window.ImageTracer = ImageTracer
  }

  // componentWillReceiveProps(nextProps) {
  //   if (this.imageWasUploaded(nextProps)) {
  //     var url = nextProps.images.results[0].url
  //     var name = nextProps.images.results[0].filename + '.png'
  //   }
  // }
  //
  // imageWasUploaded(nextProps) {
  //   return this.props.results != nextProps.results
  // }

  showDetails(region) {
    region++
    if (region == 3)
      region = 4
    this.props.selectRegion(region)
  }

  showTumor(paths) {
    let target = document.querySelector(`#all-regions svg`)
    for(let i = 0; i < 3; i++) {
      ImageTracer.imageToSVG(
        `res/output/${i}_Brats17_2013_10_1_combined.nx.77.png`,
        (svgstr) => {
          ImageTracer.appendSVGString(svgstr, `region-${i}`)
          let paths = document.querySelectorAll(`#region-${i} svg path`)
          for (let j = 0; j < paths.length; j++) {
            if (paths[j].getAttribute('fill') == "rgb(0,0,0)") {
              continue
            }
            target.appendChild(paths[j])
            paths[j].addEventListener('click', () => this.showDetails(i))
          }
          document.querySelector(`#region-${i} svg`).remove()
        },
        'Randomsampling2',
      )
    }
  }

  render() {
    let imagePreview = null, region1 = null, region2 = null, region3 = null
    if (this.props.loading) {
      imagePreview = (
        <div className="previewText">
          <img src="res/loading.gif" />
          <div>Looking for the tumor...</div>
        </div>)
    } else if (this.props.results && this.props.results.length > 0) {
      imagePreview = (<img weidth={IMG_SIZE} height={IMG_SIZE} src={`data:image/png;base64,${this.props.results[0].url}`} />)
      region1 = (<img weidth={IMG_SIZE} height={IMG_SIZE} src={`data:image/png;base64,${this.props.results[1].url}`} />)
      region2 = (<img weidth={IMG_SIZE} height={IMG_SIZE} src={`data:image/png;base64,${this.props.results[2].url}`} />)
      region3 = (<img weidth={IMG_SIZE} height={IMG_SIZE} src={`data:image/png;base64,${this.props.results[3].url}`} />)
    } else {
      imagePreview = (<div className="previewText">Please upload an Image</div>)
    }
    return (
      <div className='brain-container' ref={(container) => { this.container = container }}>
        <div className="region" id="brain">{imagePreview}</div>
        <div className="region" id="region-0">{region1}</div>
        <div className="region" id="region-1">{region2}</div>
        <div className="region" id="region-2">{region3}</div>
      </div>
    )
    /*
    <div className="region" id="all-regions">
      <svg width="256" height="256" version="1.1" xmlns="http://www.w3.org/2000/svg" desc="Created with imagetracer.js version 1.2.0" viewBox="0 0 128 128" preserveAspectRatio="none">
      </svg>
    </div>
    */
  }
}

BrainView.propTypes = {
  selectRegion: PropTypes.func.isRequired,
  results: PropTypes.array,
  loading: PropTypes.bool.isRequired,
}

function mapStateToProps(state) {
  return {
    results: state.images.results,
    loading: state.status.loading,
  }
}

function mapDispatchToProps(dispatch) {
  return {
    selectRegion: (region) => dispatch(selectRegion(region)),
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(BrainView)
