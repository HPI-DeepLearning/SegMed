import React, { Component } from 'react';

export default class LandingPage extends Component {
  render() {
    return (
      <div className='land'>
        <div className='header'>
          <h1 className='headLine'>Tumor Viz</h1>
        </div>
        <div className='description'>
          <h3 className='descriptionLine'>To analyze your brain just add a picture of your brain :)</h3>
        </div>
        <div className='content'>
          <div className='addContainer'>
            <a className='icon'>
              <i className="fa fa-plus" aria-hidden="true"></i>
              <p>Add a picture</p>
            </a>
          </div>
        </div>
      </div>
    );
  }
}
