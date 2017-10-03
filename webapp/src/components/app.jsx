import React, { Component } from 'react';

import LandingPage from './LandingPage'
import AnalyzingPage from './AnalyzingPage'

export default class App extends Component {
  render() {
    return (
      <div className='app col-lg-12'>
        <AnalyzingPage />
      </div>
    );
  }
}
