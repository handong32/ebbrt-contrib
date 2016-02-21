#include "EbbRTReconstruction.h"

EBBRT_PUBLISH_TYPE(, EbbRTReconstruction);

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

void EbbRTReconstruction::Print(ebbrt::Messenger::NetworkId nid, const char* str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char*>(buf->MutData()), len, "%s", str);

  std::cout << "EbbRTReconstruction length of sent iobuf: "
            << buf->ComputeChainDataLength() << " bytes" << std::endl;

  SendMessage(nid, std::move(buf));
}

void EbbRTReconstruction::ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                                    std::unique_ptr<ebbrt::IOBuf>&& buffer) {

  auto output = std::string(reinterpret_cast<const char*>(buffer->Data()));
  std::cout << "Received ip: " << nid.ToString() << std::endl;
  std::cout << output << std::endl;
  ebbrt::event_manager->ActivateContext(std::move(*emec));
}

void EbbRTReconstruction::runRecon() {
    // get the event manager context and save it
    ebbrt::EventManager::EventContext context;
    
    //irtkImageAttributes attr = reconstructor->_reconstructed.GetImageAttributes();
    size_t max_slices = reconstructor->_slices.size();
    size_t inputIndex = 0;
    
    int start, end, factor;

    // all sizes are the same
    std::cout << "_slices.size() " << reconstructor->_slices.size() << std::endl;
    std::cout << "_transformations.size() "
	      << reconstructor->_transformations.size() << std::endl;
    
    for (int i = 0; i < (int)nids.size(); i++) {
	// serialization
	std::ostringstream ofs;
	boost::archive::text_oarchive oa(ofs);
	
	std::cout << "Before serialize" << std::endl;
	
	//_slices
	factor = (int)ceil(max_slices / (float)numNodes);
	start = i * factor;
	end = i * factor + factor;
	end = ((size_t)end > max_slices) ? max_slices : end;

	std::cout << "start = " << start << " end = " << end << std::endl;
	
	oa& start& end;
	
	for (int j = start; j < end; j++) {
	    oa& reconstructor->_slices[j];
	}
	
	for (int j = start; j < end; j++) {
	    oa& reconstructor->_transformations[j];
	}

	for (int j = start; j < end; j++) {
	    oa& reconstructor->_simulated_slices[j];
	}

	for (int j = start; j < end; j++) {
	    oa& reconstructor->_simulated_weights[j];
	}

	for (int j = start; j < end; j++) {
	    oa& reconstructor->_simulated_inside[j];
	}

	oa& reconstructor->_reconstructed& reconstructor->_mask &max_slices & reconstructor->_global_bias_correction;
	
	std::string ts = "E " + ofs.str();
	
	std::cout << "Sending to .. " << nids[i].ToString()
		  << " size: " << ts.length() << std::endl;
	Print(nids[i], ts.c_str());
    }

    std::cout << "Saving context " << std::endl;

    emec = &context;
    ebbrt::event_manager->SaveContext(*emec);
    
    std::cout << "Received back " << std::endl;
    std::cout << "EbbRTReconstruction done " << std::endl;
    ebbrt::active_context->io_service_.stop();
}
    
