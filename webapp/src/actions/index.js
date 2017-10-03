export const startedUpload = () => {
  return {
    type: 'STARTED_UPLOAD',
  }
}

export const finishedUpload = () => {
  return {
    type: 'FINISHED_UPLOAD',
  }
}

export const receivedResults = (data) => {
  return {
    type: 'RECEIVED_RESULTS',
    payload: {
      data,
    }
  }
}

const getTitle = (regionNumber) => {
  if (regionNumber == 1) {
    return 'Whole tumor'
  } else if (regionNumber == 2) {
    return 'Non-enhancing solid tumor core'
  } else if (regionNumber == 3) {
    return 'Enhancing tumor structures'
  } else if (regionNumber == 4) {
    return 'Cystic/necrotic components of the core'
  }
  return 'Select a tumor region of your interest'
}

export const selectRegion = (region) => {
  return {
    type: 'SELECT_REGION',
    payload: {
      region,
      regionTitle: getTitle(region),
    }
  }
}
